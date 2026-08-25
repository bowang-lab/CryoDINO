#!/usr/bin/env python
"""Build a membrain data_dir from a MONAI-style datalist JSON (paths used verbatim).

JSON: {"training":[{image,label}], "validation":[...], "test":[...]}. Entries may be .pt cubes
or .nii.gz/.mrc volumes (auto-detected). Labels are binarised (>0 -> 1).

Writes:
  <data_dir>/imagesTr/<X>_0000.nii.gz  labelsTr/<X>.nii.gz   (from 'training')
  <data_dir>/imagesVal/<X>_0000.nii.gz labelsVal/<X>.nii.gz  (from 'validation')
  <test_out>  = the JSON 'test' list, copied verbatim, for the inference step
where X = basename of the entry's LABEL with extension stripped (membrain pairing rule:
label X.nii.gz <-> image X_0000.nii.gz).
"""
import argparse
import json
import os

import nibabel as nib
import numpy as np

from datalist_utils import binarize, load_datalist, load_volume, strip_ext

AFFINE = np.eye(4, dtype=np.float32)


def save_nii(arr, path):
    nib.save(nib.Nifti1Image(np.ascontiguousarray(arr), AFFINE), path)


def convert_split(entries, img_dir, lab_dir):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    for i, e in enumerate(entries, 1):
        X = strip_ext(e["label"])
        img = load_volume(e["image"]).astype(np.float32)
        seg = binarize(load_volume(e["label"]))
        save_nii(img, os.path.join(img_dir, f"{X}_0000.nii.gz"))
        save_nii(seg, os.path.join(lab_dir, f"{X}.nii.gz"))
        print(f"  [{i}/{len(entries)}] {X:32s} fg={int(seg.sum())}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", required=True)
    ap.add_argument("--data-dir", required=True, help="output membrain data_dir")
    ap.add_argument("--test-out", required=True, help="path to write the test list JSON")
    ap.add_argument("--dry-run", action="store_true", help="just print the resolved file lists")
    args = ap.parse_args()

    d = load_datalist(args.datalist)
    print(f"datalist: training={len(d['training'])} validation={len(d['validation'])} "
          f"test={len(d['test'])}")
    if args.dry_run:
        for split in ("training", "validation", "test"):
            print(f"\n[{split}]")
            for e in d[split]:
                print(f"  {strip_ext(e['label']):32s} img={e['image']}")
        return

    print("== training -> imagesTr/labelsTr ==")
    convert_split(d["training"], os.path.join(args.data_dir, "imagesTr"),
                  os.path.join(args.data_dir, "labelsTr"))
    print("== validation -> imagesVal/labelsVal ==")
    convert_split(d["validation"], os.path.join(args.data_dir, "imagesVal"),
                  os.path.join(args.data_dir, "labelsVal"))

    os.makedirs(os.path.dirname(os.path.abspath(args.test_out)), exist_ok=True)
    with open(args.test_out, "w") as f:
        json.dump(d["test"], f, indent=2)
    print(f"\ndata_dir ready: {args.data_dir}\ntest list ({len(d['test'])}) -> {args.test_out}")


if __name__ == "__main__":
    main()
