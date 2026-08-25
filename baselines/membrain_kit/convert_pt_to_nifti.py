"""Convert .pt patch tensors into a membrain-seg data_dir (binary segmentation).

Source layout (one .pt per patch, matched by filename between the two folders):
    <src>/images/<name>.pt   -> density volume (float tensor, any 3D shape)
    <src>/labels/<name>.pt   -> integer class ids stored as a tensor (0 = background)

Produces:
    <dst>/imagesTr/<name>_0000.nii.gz   <dst>/labelsTr/<name>.nii.gz
    <dst>/imagesVal/<name>_0000.nii.gz  <dst>/labelsVal/<name>.nii.gz

Key rules baked in (learned the hard way):
  * membrain pairing: image filename = label-basename + "_0000.nii.gz".
  * Labels are BINARISED: foreground = (label > 0), background = 0. (Multi-class needs code
    changes in membrain; not supported here.)
  * Padding is kept as background 0, NOT membrain's ignore label 2. Marking padding as ignore
    lets an all-padding random crop become 100% ignore -> Dice/CE loss divides by zero -> NaN.
  * Train/val split is by GROUP (default: filename with the `_patch_...` suffix stripped), so
    whole tomograms stay on one side -> no spatial leakage. Deterministic (sorted).
"""
import argparse
import os
import re

import nibabel as nib
import numpy as np
import torch

AFFINE = np.eye(4, dtype=np.float32)  # spacing 1.0; only relative geometry matters for training


def group_of(name, regex):
    return re.sub(regex, "", name)


def save_nii(arr, path):
    nib.save(nib.Nifti1Image(np.ascontiguousarray(arr), AFFINE), path)


def convert_one(name, src, dst, split):
    img = torch.load(os.path.join(src, "images", name + ".pt"), map_location="cpu").numpy()
    lab = torch.load(os.path.join(src, "labels", name + ".pt"), map_location="cpu").numpy()
    seg = (lab > 0).astype(np.uint8)  # binarize; padding stays background
    img_dir = os.path.join(dst, "imagesTr" if split == "train" else "imagesVal")
    lab_dir = os.path.join(dst, "labelsTr" if split == "train" else "labelsVal")
    save_nii(img.astype(np.float32), os.path.join(img_dir, f"{name}_0000.nii.gz"))
    save_nii(seg, os.path.join(lab_dir, f"{name}.nii.gz"))
    return int((seg == 1).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="dir with images/ and labels/ subfolders of .pt")
    ap.add_argument("--dst", required=True, help="output membrain data_dir")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of GROUPS held out for validation (default 0.2)")
    ap.add_argument("--group-regex", default=r"_patch_.*$",
                    help="regex removed from a filename to get its group id (default '_patch_.*$')")
    ap.add_argument("--val-groups", default="",
                    help="comma-separated group ids to force into val (overrides --val-frac)")
    ap.add_argument("--limit", type=int, default=0, help="convert only N patches (dry-run)")
    args = ap.parse_args()

    for sub in ("imagesTr", "labelsTr", "imagesVal", "labelsVal"):
        os.makedirs(os.path.join(args.dst, sub), exist_ok=True)

    names = sorted(f[:-3] for f in os.listdir(os.path.join(args.src, "images"))
                   if f.endswith(".pt"))
    if args.limit:
        names = names[: args.limit]

    groups = sorted({group_of(n, args.group_regex) for n in names})
    if args.val_groups.strip():
        val_groups = {g.strip() for g in args.val_groups.split(",") if g.strip()}
    else:
        n_val = max(1, round(args.val_frac * len(groups)))
        val_groups = set(groups[-n_val:])  # deterministic: last N groups after sorting
    print(f"{len(groups)} groups, {len(val_groups)} -> val: {sorted(val_groups)}")

    n_tr = n_val = 0
    for i, name in enumerate(names, 1):
        split = "val" if group_of(name, args.group_regex) in val_groups else "train"
        fg = convert_one(name, args.src, args.dst, split)
        n_tr += split == "train"
        n_val += split == "val"
        print(f"[{i}/{len(names)}] {name:32s} -> {split:5s} fg={fg}", flush=True)

    print(f"\nDone. train={n_tr} val={n_val} patches  ->  {args.dst}")


if __name__ == "__main__":
    main()
