#!/usr/bin/env python
"""Build a small multi-class mock dataset by cropping real Dataset010 patches.

Dataset010 labels already carry three genuine particle classes {1,2,3}, so the mock is real data,
just tiny -- which keeps per-class Dice meaningful instead of measuring noise. Patches whose best
window does not contain every class are skipped (the edge patches hold only class 1).

    python tests/make_mock_multiclass.py --out <dir> [--crop 128] [--n 5]

Writes <out>/images/<name>.pt, <out>/labels/<name>.pt and <out>/mock_mc_datalist.json in the same
shape as a real datalist, so nothing downstream is special-cased.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

SRC = ("/home/cyyu/projects/def-wanglab-ab/cryodino/010/"
       "Dataset010_CZII_10010_train_patches_512")


def best_window(lab, crop, stride, classes):
    """Window maximising the least-represented class; None if no window has them all."""
    best, best_min = None, 0
    for z in range(0, lab.shape[0] - crop + 1, stride):
        for y in range(0, lab.shape[1] - crop + 1, stride):
            for x in range(0, lab.shape[2] - crop + 1, stride):
                w = lab[z:z + crop, y:y + crop, x:x + crop]
                per = [int((w == c).sum()) for c in classes]
                if min(per) > best_min:
                    best_min, best = min(per), (z, y, x, per)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--n", type=int, default=5, help="tomograms to produce (last one is test)")
    ap.add_argument("--classes", default="1,2,3", help="label values that must all be present")
    args = ap.parse_args()

    classes = [int(c) for c in args.classes.split(",")]
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)

    names = sorted(f[:-3] for f in os.listdir(os.path.join(args.src, "labels"))
                   if f.endswith(".pt"))
    entries, skipped = [], []
    for name in names:
        if len(entries) >= args.n:
            break
        lab = torch.load(os.path.join(args.src, "labels", name + ".pt"),
                         map_location="cpu").numpy()
        win = best_window(lab, args.crop, args.stride, classes)
        if win is None:
            skipped.append(name)
            continue
        z, y, x, per = win
        img = torch.load(os.path.join(args.src, "images", name + ".pt"),
                         map_location="cpu").numpy()
        sl = (slice(z, z + args.crop), slice(y, y + args.crop), slice(x, x + args.crop))
        mock = f"MOCK_{len(entries)}_{name}"
        ip = os.path.abspath(os.path.join(args.out, "images", mock + ".pt"))
        lp = os.path.abspath(os.path.join(args.out, "labels", mock + ".pt"))
        torch.save(torch.from_numpy(np.ascontiguousarray(img[sl])), ip)
        torch.save(torch.from_numpy(np.ascontiguousarray(lab[sl])), lp)
        entries.append({"image": ip, "label": lp})
        print(f"  {mock:44s} window=({z},{y},{x})  per-class={dict(zip(classes, per))}", flush=True)

    if len(entries) < 2:
        raise SystemExit(f"only {len(entries)} usable windows found -- need at least 2")
    if skipped:
        print(f"\nskipped {len(skipped)} patch(es) with no all-classes window: "
              f"{', '.join(skipped[:4])}{' ...' if len(skipped) > 4 else ''}")

    d = {"training": entries[:-1], "validation": [], "test": entries[-1:]}
    dl = os.path.join(args.out, "mock_mc_datalist.json")
    json.dump(d, open(dl, "w"), indent=2)
    print(f"\ntrain={len(d['training'])} test={len(d['test'])}\ndatalist -> {dl}")


if __name__ == "__main__":
    sys.exit(main())
