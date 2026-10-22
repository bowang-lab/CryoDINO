#!/usr/bin/env python
"""Score DeepETPicker on the benchmark split exactly as detection3d.py scores CryoDINO.

Same metric code: CZIDetectionMetrics from 3DINO/dinov2/eval/detection_3d/metrics.py (CZII
radii, 0.5 x radius matching, beta = 4, class weights). Same protocol: per-class score
thresholds are chosen on the validation tomograms and applied once to the test tomograms; the
test set is never swept.

DeepETPicker writes no confidence, only a normalised cluster size ("area", 5th column of
Coords_withArea/<tomo>.coords), so that is the score being thresholded. The candidate
thresholds per class are 0 (keep everything) plus quantiles of that class's val scores.

Writes <pred-dir>/results.json with the detection3d.py keys (test_f4, test_per_cls_f4,
best_val_f4, best_val_thresholds) plus val_per_cls_f4 and counts.
"""
import argparse
import importlib.util
import json
import os

import numpy as np

from datalist_utils import read_coords

METRICS_PY = "3DINO/dinov2/eval/detection_3d/metrics.py"


def load_metrics_module(repo):
    # Load the file directly: importing the dinov2 package would pull in torch/xformers config.
    spec = importlib.util.spec_from_file_location("cryodino_det_metrics", os.path.join(repo, METRICS_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def accumulate(metric, names, pred_dir, gt_dir, voxel_size, num_cls):
    """Feed every tomogram of one role into a CZIDetectionMetrics; return its pooled scores."""
    scores_by_cls = {c: [] for c in range(num_cls)}
    n_missing = 0
    for name in names:
        path = os.path.join(pred_dir, name + ".coords")
        if not os.path.exists(path):
            n_missing += 1
        pred = read_coords(path)                         # (K, 5): cls+1, x, y, z, area
        pred = pred[(pred[:, 0] >= 1) & (pred[:, 0] <= num_cls)]
        labels = pred[:, 0].astype(int) - 1
        scores = pred[:, 4] if pred.shape[1] > 4 else np.ones(len(pred))
        gt = np.load(os.path.join(gt_dir, name + ".npy"))        # (N, 4): x, y, z, cls
        gt_points = np.column_stack([gt[:, :4], np.zeros(len(gt))])  # [x, y, z, cls, r]
        metric.accumulate(pred[:, 1:4], labels, scores, gt_points, voxel_size)
        for c, s in zip(labels, scores):
            scores_by_cls[c].append(s)
    if n_missing:
        raise SystemExit(f"{n_missing}/{len(names)} tomograms have no prediction file in {pred_dir}")
    return scores_by_cls


def sweep_val(metric, scores_by_cls, n_quantiles):
    """Per-class best threshold on val, mirroring CZIDetectionMetrics.threshold_sweep."""
    names = metric.PARTICLE_NAMES
    best_f4 = {n: 0.0 for n in names}
    best_thr = {n: 0.0 for n in names}
    candidates = {0.0}
    for s in scores_by_cls.values():
        if s:
            candidates.update(np.quantile(s, np.linspace(0, 1, n_quantiles)).tolist())
    for thr in sorted(candidates):
        _, per_class = metric._f4_at_threshold(float(thr))
        for n, f4 in per_class.items():
            if f4 > best_f4[n]:
                best_f4[n], best_thr[n] = f4, float(thr)
    agg = sum(best_f4[n] * w for n, w in zip(names, metric.WEIGHTS)) / sum(metric.WEIGHTS)
    return agg, best_thr, best_f4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base_path from prepare_benchmark.py")
    ap.add_argument("--pred-dir", required=True, help="PredictedLabels/ (or its Coords_withArea/)")
    ap.add_argument("--repo", required=True, help="CryoDINO checkout (for metrics.py)")
    ap.add_argument("--n-quantiles", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pred_dir = args.pred_dir
    if os.path.isdir(os.path.join(pred_dir, "Coords_withArea")):
        pred_dir = os.path.join(pred_dir, "Coords_withArea")
    with open(os.path.join(args.base, "dataset_split.json")) as f:
        split = json.load(f)
    roles, vs, num_cls = split["roles"], split["voxel_size"], split["num_cls"]
    val_names = sorted(n for n, r in roles.items() if r == "val")
    test_names = sorted(n for n, r in roles.items() if r == "test")
    gt_dir = os.path.join(args.base, "gt")

    mod = load_metrics_module(args.repo)
    val_metric, test_metric = mod.CZIDetectionMetrics(), mod.CZIDetectionMetrics()
    if num_cls != len(val_metric.PARTICLE_NAMES):
        raise SystemExit(f"{num_cls} classes in the split vs {len(val_metric.PARTICLE_NAMES)} in metrics.py")

    val_scores = accumulate(val_metric, val_names, pred_dir, gt_dir, vs, num_cls)
    best_val_f4, best_thr, val_per_cls = sweep_val(val_metric, val_scores, args.n_quantiles)
    accumulate(test_metric, test_names, pred_dir, gt_dir, vs, num_cls)
    test_f4, test_per_cls = test_metric.f4_at_thresholds(best_thr)

    print(f"val  ({len(val_names)} tomograms) best F4 {best_val_f4:.4f}  per class {val_per_cls}")
    print(f"val-selected thresholds (area): {best_thr}")
    print(f"test ({len(test_names)} tomograms) F4 @ val thresholds {test_f4:.4f}  per class {test_per_cls}")

    out = args.out or os.path.join(os.path.dirname(pred_dir.rstrip("/")), "results.json")
    with open(out, "w") as f:
        json.dump({
            "test_f4": float(test_f4),
            "test_per_cls_f4": {k: float(v) for k, v in test_per_cls.items()},  # at val thresholds
            "best_val_f4": float(best_val_f4),
            "best_val_thresholds": best_thr,
            "val_per_cls_f4": {k: float(v) for k, v in val_per_cls.items()},
            "score": "DeepETPicker normalised cluster area (5th column of Coords_withArea)",
            "n_val_tomograms": len(val_names),
            "n_test_tomograms": len(test_names),
        }, f, indent=2)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
