import os
import argparse
from datetime import datetime
from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm

from exposure_adjuster import ExposureAdjuster
from my_utils.training_utils import MEFtoMESDataset


@torch.no_grad()
def encode_image(model: ExposureAdjuster, c_t: torch.Tensor):
    """
    Run the ev_norm-independent part of the forward pass once per image:
      1) Haar DWT -> LL, LH, HL, HH
      2) WaveletFusionNet -> fused [B, 3, H, W] in [0, 1]
      3) Diffusion (VAE encode -> UNet 1-step -> VAE decode) -> diff_out [-1, 1]
      4) Map diff_out to [0, 1]

    Returns:
        diff_out_01: [B, 3, H, W] in [0, 1]
    """
    # Step 1-2: wavelet fusion
    fused = model.get_fused(c_t)  # [B, 3, H, W] in [0, 1]

    # Step 3: diffusion
    B = c_t.shape[0]
    caption_enc = model.learnable_prompt.expand(B, -1, -1).to(dtype=c_t.dtype)

    c_latent = model.vae.encode(fused).latent_dist.sample()
    c_latent = c_latent * model.vae.config.scaling_factor

    model_pred = model.unet(
        c_latent,
        model.timesteps,
        encoder_hidden_states=caption_enc,
    ).sample

    x_denoised = model.sched.step(
        model_pred, model.timesteps, c_latent, return_dict=True
    ).prev_sample

    model.vae.decoder.incoming_skip_acts = model.vae.encoder.current_down_blocks
    diff_out = model.vae.decode(
        x_denoised / model.vae.config.scaling_factor
    ).sample.clamp(-1.0, 1.0)

    # Step 4: remap to [0, 1]
    return (diff_out + 1.0) * 0.5


@torch.no_grad()
def decode_with_ev(model: ExposureAdjuster, diff_out_01: torch.Tensor, c_t_01: torch.Tensor, ev: float):
    """
    Apply wavelet-domain tone mapping for a given target EV.

    Args:
        diff_out_01: [B, 3, H, W] in [0, 1]  -- output of encode_image()
        c_t_01:      [B, 3, H, W] in [0, 1]  -- original input image
        ev:          float in [-3, +3]        -- target exposure value (EV)

    Returns:
        out: [B, 3, H, W] in [-1, 1]
    """
    device = diff_out_01.device
    dtype  = diff_out_01.dtype
    B      = diff_out_01.shape[0]

    # Normalise EV from [-3, +3] to [0, 1], matching MEFtoMESDataset._ev_to_norm
    ev_norm_val = (ev - MEFtoMESDataset._EV_MIN) / (MEFtoMESDataset._EV_MAX - MEFtoMESDataset._EV_MIN)
    ev_norm = torch.full((B, 1), ev_norm_val, device=device, dtype=dtype)

    out_01 = model.tone_mapping(diff_out_01, c_t_01, ev_norm)  # [0, 1]  ← 补上 c_t_01
    return torch.clamp(out_01 * 2.0 - 1.0, -1.0, 1.0)


def ev_file_tag(ev: int) -> str:
    """
    Build a filename tag that sorts correctly (-3 -> +3) under lexicographic order.

    Format: ev{offset:02d}({sign}{abs})
      offset = ev + 3  (maps [-3,+3] -> [00,06])
      e.g.  -3 -> "ev00(-3)"
            -1 -> "ev02(-1)"
             0 -> "ev03(0)"
            +2 -> "ev05(+2)"
            +3 -> "ev06(+3)"
    """
    offset   = ev + 3                                    # 0 .. 6
    readable = f"{ev:+d}" if ev != 0 else "0"
    return f"ev{offset:02d}({readable})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  type=str, required=True,
                        help="Path to input image directory")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint (.pkl)")
    parser.add_argument("--output_dir", type=str, default="result",
                        help="Directory to save outputs")
    args = parser.parse_args()

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    model = ExposureAdjuster(pretrained_path=args.model_path)
    model.set_eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # EV values in ascending order so outputs are saved -3 -> +3
    ev_list   = [-3, -2, -1, 0, 1, 2, 3]
    to_pil    = transforms.ToPILImage()
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for filename in tqdm(sorted(os.listdir(args.input_dir))):
        if not filename.lower().endswith(valid_ext):
            continue

        input_image = Image.open(os.path.join(args.input_dir, filename)).convert("RGB")

        # Crop to multiple of 8 (required by VAE and wavelet DWT)
        new_w = input_image.width  - input_image.width  % 8
        new_h = input_image.height - input_image.height % 8
        if new_w == 0 or new_h == 0:
            raise ValueError(f"Image {filename} is too small after multiple-of-8 adjustment.")
        if (new_w, new_h) != input_image.size:
            input_image = input_image.resize((new_w, new_h), Image.LANCZOS)

        c_t = F.to_tensor(input_image).unsqueeze(0).to(device)

        # Heavy part: run once per image
        diff_out_01 = encode_image(model, c_t)

        # Light part: one tone-mapping pass per target EV
        name, ext = os.path.splitext(filename)
        for ev in ev_list:
            out = decode_with_ev(model, diff_out_01, c_t, ev)[0].cpu()
            to_pil(out * 0.5 + 0.5).save(
                os.path.join(output_dir, f"{name}_{ev_file_tag(ev)}{ext}")
            )