"""
build_mess_manifest.py
======================
Scan a directory of pre-rendered RAW scenes and write the JSONL manifest that
``MESSTripletDataset`` consumes.

Expected layout (one folder per RAW scene, any depth of <subset> grouping):

    <data_root>/
        <subset>/                       e.g. fivek, AODRAW, RAISE, ...
            <scene>/
                style/                  style-varied renderings T_sty(X; s)
                    mef_mertens.png       (MEF, tone mapping, different ISP styles...)
                    tm_reinhard.png
                    ...
                exposure/               camera-consistent renderings T_cam(X; e)
                    ev-3.png  ev-2.png  ev-1.png  ev+0.png  ev+1.png  ev+2.png  ev+3.png
                    (any name containing the EV works: "-1.5.png", "ev_+0.5.jpg", ...)

Usage:
    python tools/build_mess_manifest.py --data_root /data/mess_raw --out /data/mess_raw/train.jsonl
    python tools/build_mess_manifest.py --data_root /data/mess_raw --out val.jsonl --subsets fivek --max_scenes 50
"""

import argparse
import json
import os
import re

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
EV_TAG_RE = re.compile(r"ev[_\s]*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)   # "...ev+1.5..."
EV_NUM_RE = re.compile(r"\s*([+-]?\d+(?:\.\d+)?)\s*")                       # stem is just "-1.5"


def parse_ev(filename):
    stem = os.path.splitext(filename)[0]
    m = EV_TAG_RE.findall(stem)
    if m:
        return float(m[-1])
    m = EV_NUM_RE.fullmatch(stem)
    if m:
        return float(m.group(1))
    raise ValueError(f"Cannot parse EV from file name: {filename}")


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT)) if os.path.isdir(d) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--style_dir", default="style")
    ap.add_argument("--exposure_dir", default="exposure")
    ap.add_argument("--subsets", nargs="*", default=None, help="Only include these top-level subsets.")
    ap.add_argument("--max_scenes", type=int, default=None)
    ap.add_argument("--absolute", action="store_true", help="Write absolute paths.")
    args = ap.parse_args()

    root = os.path.abspath(args.data_root)
    n_scene, n_skip = 0, 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for dirpath, dirnames, _ in os.walk(root):
            dirnames.sort()
            if os.path.basename(dirpath) == args.exposure_dir or os.path.basename(dirpath) == args.style_dir:
                continue
            exp_dir = os.path.join(dirpath, args.exposure_dir)
            if not os.path.isdir(exp_dir):
                continue
            rel_scene = os.path.relpath(dirpath, root)
            if args.subsets and rel_scene.split(os.sep)[0] not in args.subsets:
                continue

            mk = (lambda p: p) if args.absolute else (lambda p: os.path.relpath(p, root))
            exposure = []
            for f in list_images(exp_dir):
                try:
                    exposure.append({"ev": parse_ev(f), "path": mk(os.path.join(exp_dir, f))})
                except ValueError as e:
                    print(f"[warn] {rel_scene}: {e}")
            style = [mk(os.path.join(dirpath, args.style_dir, f))
                     for f in list_images(os.path.join(dirpath, args.style_dir))]

            if not exposure:
                n_skip += 1
                continue
            exposure.sort(key=lambda e: e["ev"])
            fout.write(json.dumps({"id": rel_scene.replace(os.sep, "/"), "style": style,
                                   "exposure": exposure}, ensure_ascii=False) + "\n")
            n_scene += 1
            if args.max_scenes and n_scene >= args.max_scenes:
                break

    print(f"Wrote {n_scene} scenes to {args.out} (skipped {n_skip}).")


if __name__ == "__main__":
    main()
