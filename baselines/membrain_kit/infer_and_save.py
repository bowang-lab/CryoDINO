#!/usr/bin/env python
"""Run membrain inference on every test sample and save one output per sample.

For each entry in the test list: convert the image to .mrc (membrain's `segment` reads MRC
only), run `membrain segment`, and save results under <out>/test_predictions/<X>/:
  <X>.mrc                  the input (mrc)
  <X>_*_segmented.mrc      the binary segmentation  (membrain output)
  <X>_*_scores.mrc         the probability map       (--store-probabilities)
If the entry has a label, binarise it, compute Dice vs the segmentation, and append to
<out>/test_predictions/metrics.csv.
"""
import argparse
import csv
import glob
import json
import os
import subprocess

import numpy as np

from datalist_utils import binarize, load_volume, save_mrc, strip_ext


def dice(a, b):
    a = a > 0
    b = b > 0
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(a, b).sum() / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-list", required=True, help="JSON list of {'image','label'} entries")
    ap.add_argument("--ckpt", required=True, help="trained membrain .ckpt")
    ap.add_argument("--out", required=True, help="run output dir; predictions go under test_predictions/")
    ap.add_argument("--threshold", type=float, default=0.0, help="segmentation threshold (logits)")
    args = ap.parse_args()

    with open(args.test_list) as f:
        test = json.load(f)
    pred_root = os.path.join(args.out, "test_predictions")
    os.makedirs(pred_root, exist_ok=True)
    rows = []

    for i, e in enumerate(test, 1):
        X = strip_ext(e.get("label") or e["image"])
        d = os.path.join(pred_root, X)
        os.makedirs(d, exist_ok=True)
        # membrain segment reads MRC only -> materialize the image as .mrc
        img_mrc = e["image"] if e["image"].endswith((".mrc", ".rec")) else os.path.join(d, X + ".mrc")
        if img_mrc != e["image"]:
            save_mrc(load_volume(e["image"]), img_mrc)
        print(f"[{i}/{len(test)}] segmenting {X} ...", flush=True)
        subprocess.run(
            ["membrain", "segment", "--tomogram-path", img_mrc, "--ckpt-path", args.ckpt,
             "--out-folder", d, "--no-rescale-patches", "--store-probabilities",
             "--segmentation-threshold", str(args.threshold)],
            check=True,
        )
        seg_files = sorted(glob.glob(os.path.join(d, "*_segmented.mrc")))
        seg_path = seg_files[-1] if seg_files else ""
        row = {"name": X, "segmentation": seg_path}
        if e.get("label") and seg_path:
            gt = binarize(load_volume(e["label"]))
            pred = load_volume(seg_path)
            row["dice"] = round(dice(pred, gt), 4) if pred.shape == gt.shape else "shape_mismatch"
        rows.append(row)
        print(f"    -> {seg_path}  dice={row.get('dice','n/a')}", flush=True)

    if any("dice" in r for r in rows):
        with open(os.path.join(pred_root, "metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "dice", "segmentation"])
            w.writeheader()
            for r in rows:
                w.writerow({"name": r["name"], "dice": r.get("dice", ""), "segmentation": r["segmentation"]})
        print(f"\nmetrics.csv -> {os.path.join(pred_root, 'metrics.csv')}")
    print(f"saved {len(rows)} test prediction(s) under {pred_root}")


if __name__ == "__main__":
    main()
