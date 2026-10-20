"""
mess_dataset.py
===============
Data provider for MESS training (paper Sec. 3.1 / Appendix A).

Each RAW scene X yields a *group* of pre-rendered RGB images (RAW processing is
assumed to be done offline; this module only consumes the renders):

    * style pool      {T_sty(X; s)}   -- MEF / tone-mapping / direct renders with
                                         diverse rendering styles.  Two of them are
                                         sampled per iteration as (I_i, I_j).
    * exposure set    {T_cam(X; e)}   -- camera-consistent renders at known EVs.
                                         One is sampled as the target I_t*.

Per sample the dataset returns the RAW-guided triplet {I_i, I_j, I_t*} plus the
normalised target exposure e_t in [0, 1].

---------------------------------------------------------------------------
On-disk format
---------------------------------------------------------------------------
The dataset is driven by a JSONL *manifest*; one line = one RAW scene:

    {"id": "fivek/a0001",
     "style":    ["fivek/a0001/style/mef_mertens.png",
                  "fivek/a0001/style/tm_reinhard.png", ...],
     "exposure": [{"ev": -3.0, "path": "fivek/a0001/exposure/ev-3.png"},
                  {"ev":  0.0, "path": "fivek/a0001/exposure/ev+0.png"}, ...]}

Relative paths are resolved against ``data_root``.  A manifest can be generated
from the recommended directory layout with ``tools/build_mess_manifest.py``:

    <data_root>/<subset>/<scene>/style/*.{png,jpg,...}
    <data_root>/<subset>/<scene>/exposure/ev<+/-x.y>.{png,jpg,...}

All renders of the same scene must be pixel-aligned (same RAW, same geometry).
8-bit and 16-bit images are both supported.
"""

import json
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as nnF
from torch.utils.data import Dataset

try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:  # pragma: no cover
    cv2 = None
    from PIL import Image


# EV range used by the paper (and by inference.py through MEFtoMESDataset).
EV_MIN, EV_MAX = -3.0, 3.0


def ev_to_norm(ev, ev_min=EV_MIN, ev_max=EV_MAX):
    """Linearly map an EV value in [ev_min, ev_max] to e_t in [0, 1]."""
    return (float(ev) - ev_min) / (ev_max - ev_min)


def load_image(path):
    """Load an RGB image as float32 tensor [3, H, W] in [0, 1] (8/16-bit aware)."""
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to read image: {path}")
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = img[:, :, ::-1]  # BGR -> RGB
        scale = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
        img = img.astype(np.float32) / scale
    else:
        pil = Image.open(path)
        img = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).clamp_(0.0, 1.0)


def read_manifest(manifest_paths, data_root=None):
    """Read one or more JSONL manifests into a list of scene dicts with absolute paths."""
    if isinstance(manifest_paths, str):
        manifest_paths = [manifest_paths]
    scenes = []
    for mpath in manifest_paths:
        root = data_root if data_root is not None else os.path.dirname(os.path.abspath(mpath))
        with open(mpath, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                resolve = lambda p: p if os.path.isabs(p) else os.path.join(root, p)
                scenes.append({
                    "id": rec.get("id", f"{os.path.basename(mpath)}:{ln}"),
                    "style": [resolve(p) for p in rec.get("style", [])],
                    "exposure": [(float(e["ev"]), resolve(e["path"])) for e in rec.get("exposure", [])],
                })
    return scenes


class MESSTripletDataset(Dataset):
    """
    RAW-guided triplet dataset {I_i, I_j, I_t*} for MESS training.

    Args:
        manifest:            path (or list of paths) to JSONL manifest(s).
        data_root:           root for relative paths (default: manifest dir).
        crop_size:           random crop size (paper: 512). Must be a multiple of 8.
        ev_min, ev_max:      EV range used for normalisation (paper: [-3, 3]).
        style_include_ev:    if >= 0, camera-consistent renders with |EV| <= this
                             value are also added to the style pool (paper:
                             "in addition to directly rendered exposure variants").
                             Set to a negative number to disable.
        hflip:               random horizontal flip (applied jointly to the triplet).
        distinct_pair:       sample I_i != I_j whenever the pool has >= 2 entries.

    Returns (dict):
        img_i   : [3, h, w] in [0, 1]   source rendering I_i       (model input)
        img_j   : [3, h, w] in [0, 1]   second style rendering I_j (for L_con)
        img_t   : [3, h, w] in [0, 1]   camera-consistent target I_t*
        ev_norm : [1]       in [0, 1]   normalised target exposure e_t
        ev      : [1]                   raw EV value (for logging)
    """

    def __init__(
        self,
        manifest,
        data_root=None,
        crop_size=512,
        ev_min=EV_MIN,
        ev_max=EV_MAX,
        style_include_ev=1.0,
        hflip=True,
        distinct_pair=True,
    ):
        super().__init__()
        assert crop_size % 8 == 0, "crop_size must be a multiple of 8 (VAE + Haar DWT)."
        self.crop_size = crop_size
        self.ev_min, self.ev_max = ev_min, ev_max
        self.hflip = hflip
        self.distinct_pair = distinct_pair
        self._warned_size = False

        self.scenes = []
        n_drop = 0
        for sc in read_manifest(manifest, data_root):
            exposure = [(ev, p) for ev, p in sc["exposure"] if ev_min - 1e-6 <= ev <= ev_max + 1e-6]
            style = list(sc["style"])
            if style_include_ev is not None and style_include_ev >= 0:
                style += [p for ev, p in exposure if abs(ev) <= style_include_ev + 1e-6]
            style = list(dict.fromkeys(style))  # de-duplicate, keep order
            if len(exposure) == 0 or len(style) == 0:
                n_drop += 1
                continue
            self.scenes.append({"id": sc["id"], "style": style, "exposure": exposure})

        if len(self.scenes) == 0:
            raise RuntimeError("MESSTripletDataset: no valid scene found in manifest(s).")
        if n_drop:
            warnings.warn(f"MESSTripletDataset: dropped {n_drop} scenes without style or exposure renders.")

    def __len__(self):
        return len(self.scenes)

    # ------------------------------------------------------------------
    def _align_size(self, img, hw):
        if tuple(img.shape[-2:]) == tuple(hw):
            return img
        if not self._warned_size:
            warnings.warn("Renders of one scene differ in size; resizing to the target's size. "
                          "Renders should be pixel-aligned.")
            self._warned_size = True
        return nnF.interpolate(img[None], size=hw, mode="bicubic", align_corners=False)[0].clamp(0, 1)

    def _ensure_min_size(self, imgs):
        h, w = imgs[0].shape[-2:]
        s = self.crop_size
        if min(h, w) >= s:
            return imgs
        scale = s / min(h, w)
        nh, nw = max(s, int(round(h * scale))), max(s, int(round(w * scale)))
        return [nnF.interpolate(x[None], size=(nh, nw), mode="bicubic", align_corners=False)[0].clamp(0, 1)
                for x in imgs]

    def _joint_random_crop_flip(self, imgs):
        h, w = imgs[0].shape[-2:]
        s = self.crop_size
        top = random.randint(0, h - s)
        left = random.randint(0, w - s)
        imgs = [x[:, top:top + s, left:left + s] for x in imgs]
        if self.hflip and random.random() < 0.5:
            imgs = [x.flip(-1) for x in imgs]
        return imgs

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        sc = self.scenes[idx]

        # (I_i, I_j): two style-varied renderings from the same RAW
        pool = sc["style"]
        if self.distinct_pair and len(pool) >= 2:
            p_i, p_j = random.sample(pool, 2)
        else:
            p_i, p_j = random.choice(pool), random.choice(pool)

        # I_t*: camera-consistent rendering at a uniformly sampled target EV
        ev, p_t = random.choice(sc["exposure"])

        img_t = load_image(p_t)
        hw = img_t.shape[-2:]
        img_i = self._align_size(load_image(p_i), hw)
        img_j = self._align_size(load_image(p_j), hw)

        img_i, img_j, img_t = self._joint_random_crop_flip(self._ensure_min_size([img_i, img_j, img_t]))

        return {
            "img_i": img_i.contiguous(),
            "img_j": img_j.contiguous(),
            "img_t": img_t.contiguous(),
            "ev_norm": torch.tensor([ev_to_norm(ev, self.ev_min, self.ev_max)], dtype=torch.float32),
            "ev": torch.tensor([ev], dtype=torch.float32),
        }
