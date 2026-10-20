import os
import random
import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as nnF
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args(input_args=None):
    """Parse command-line arguments for exposure adjustment training."""
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--mes_folder",  required=True, type=str,
                        help="Root of UNICEdataset (multi-exposure sequences).")
    parser.add_argument("--mef_folder",  required=True, type=str,
                        help="Root of pseudoGT (MEF fusion results).")
    parser.add_argument("--train_image_prep", default="resize_256", type=str)

    # Logging / checkpointing
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpointing_steps", type=int, default=500)

    # Model architecture
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # Training
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")

    return parser.parse_args(input_args)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def build_tensor_transform(image_prep):
    """Build a tensor-domain image transform pipeline (input/output in [0, 1])."""

    def resize_tensor(tensor, size):
        return nnF.interpolate(
            tensor.unsqueeze(0),
            size=size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

    def center_crop_tensor(tensor, size):
        h, w = tensor.shape[1], tensor.shape[2]
        th, tw = (size, size) if isinstance(size, int) else size
        i, j = (h - th) // 2, (w - tw) // 2
        return tensor[:, i:i + th, j:j + tw]

    def random_crop_tensor(tensor, size):
        h, w = tensor.shape[1], tensor.shape[2]
        th, tw = (size, size) if isinstance(size, int) else size
        i = torch.randint(0, h - th + 1, (1,)).item()
        j = torch.randint(0, w - tw + 1, (1,)).item()
        return tensor[:, i:i + th, j:j + tw]

    def random_hflip_tensor(tensor):
        return tensor.flip(-1) if torch.rand(1) < 0.5 else tensor

    if image_prep == "resized_crop_512":
        return lambda x: center_crop_tensor(resize_tensor(x, 512), 512)
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        def transform(x):
            x = resize_tensor(x, (286, 286))
            x = random_crop_tensor(x, (256, 256))
            return random_hflip_tensor(x)
        return transform
    elif image_prep in ("resize_256", "resize_256x256"):
        return lambda x: resize_tensor(x, (256, 256))
    elif image_prep in ("resize_512", "resize_512x512"):
        return lambda x: resize_tensor(x, (512, 512))
    elif image_prep == "no_resize":
        return lambda x: x
    else:
        raise ValueError(f"Unsupported image_prep: {image_prep}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MEFtoMESDataset(Dataset):
    """
    MEF-to-MES paired dataset.

    Each sample pairs a Multi-Exposure Fusion (MEF) result with one randomly
    sampled image from the corresponding Multi-Exposure Sequence (MES).

    Directory layout expected:
        mes_folder/          (UNICEdataset root)
            <subdataset>/
                <exposure>/
                    <image_name>.jpg
        mef_folder/          (pseudoGT root)
            <subdataset>/
                <image_name>.jpg

    Returns per sample:
        img1     -- [C, H, W] in [0, 1]   (MEF fusion image  -- model input)
        ev_norm  -- [1]       in [0, 1]   (EV of the MES target, normalised from [-3,+3] to [0,1])
        img2     -- [C, H, W] in [-1, 1]  (MES target image  -- model output)
    """

    SUBDATASETS = ["AODRAW", "fivek", "HDRP", "PASCAL-RAW", "PPR10K", "RAISE", "RAW-NOD"]
    EXPOSURES   = ["0", "+1", "-1", "+2", "-2", "+3", "-3"]

    # EV range is [-3, +3]; linearly normalised to [0, 1].
    _EV_VALUES          = {"0": 0, "+1": 1, "-1": -1, "+2": 2, "-2": -2, "+3": 3, "-3": -3}
    _EV_MIN, _EV_MAX    = -3.0, 3.0

    def __init__(self, mes_folder, mef_folder, image_prep):
        """
        Args:
            mes_folder:  Root of UNICEdataset (multi-exposure sequences).
            mef_folder:  Root of pseudoGT     (MEF fusion results).
            image_prep:  Spatial transform key passed to build_tensor_transform.
        """
        super().__init__()
        self.mes_folder = mes_folder
        self.mef_folder = mef_folder
        self.transform  = build_tensor_transform(image_prep)
        self.samples    = self._collect_samples()

    @classmethod
    def _ev_to_norm(cls, ev_str: str) -> float:
        """Convert an EV string (e.g. '+2') to a normalised float in [0, 1]."""
        ev = cls._EV_VALUES[ev_str]
        return (ev - cls._EV_MIN) / (cls._EV_MAX - cls._EV_MIN)

    def _collect_samples(self):
        """
        Build a list of (mef_path, [(mes_path, ev_norm), ...]) tuples.

        Only images that have both a MEF result AND at least one MES exposure
        are included.
        """
        samples = []
        for subdataset in self.SUBDATASETS:
            mes_sub = os.path.join(self.mes_folder, subdataset)
            mef_sub = os.path.join(self.mef_folder, subdataset)
            if not os.path.isdir(mes_sub) or not os.path.isdir(mef_sub):
                continue

            # Collect all MES paths grouped by image name, storing (path, ev_norm)
            name_to_mes_paths = {}
            for exposure in self.EXPOSURES:
                exp_dir = os.path.join(mes_sub, exposure)
                if not os.path.isdir(exp_dir):
                    continue
                ev_norm = self._ev_to_norm(exposure)
                for img_name in os.listdir(exp_dir):
                    img_path = os.path.join(exp_dir, img_name)
                    if not os.path.isfile(img_path):
                        continue
                    name_to_mes_paths.setdefault(img_name, []).append((img_path, ev_norm))

            # Match each MES group with its MEF counterpart
            for img_name, mes_entries in name_to_mes_paths.items():
                mef_path = os.path.join(mef_sub, img_name)
                if os.path.isfile(mef_path):
                    samples.append((mef_path, mes_entries))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mef_path, mes_entries = self.samples[idx]

        # Input: MEF fusion image  -> [0, 1]
        mef_img = F.to_tensor(Image.open(mef_path).convert("RGB"))   # [C,H,W] in [0,1]

        # Target: one randomly sampled MES exposure  -> [-1, 1]
        mes_path, ev_norm = random.choice(mes_entries)
        mes_img  = F.to_tensor(Image.open(mes_path).convert("RGB"))  # [C,H,W] in [0,1]

        # Apply the same spatial transform to both images
        mef_t    = self.transform(mef_img).clamp(0.0, 1.0)           # [0, 1]
        mes_t_01 = self.transform(mes_img).clamp(0.0, 1.0)           # [0, 1]

        # EV of the MES target, normalised to [0, 1] (used to condition the model)
        ev_norm_t = torch.tensor([ev_norm], dtype=torch.float32)      # scalar in [0,1]

        # Convert MES target to [-1, 1] for the model output space
        mes_t_m11 = F.normalize(mes_t_01, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return {
            "img1":    mef_t,       # [C,H,W] in [0,1]   -- MEF input
            "ev_norm": ev_norm_t,   # [1]     in [0,1]   -- normalised EV condition
            "img2":    mes_t_m11,   # [C,H,W] in [-1,1]  -- MES target
        }


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------
PairedDataset = MEFtoMESDataset