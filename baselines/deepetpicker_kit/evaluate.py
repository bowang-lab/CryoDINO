#!/usr/bin/env python
"""Score DeepETPicker's predicted coordinates against the ground truth.

DeepETPicker's own scorer (utils/misc.py cal_metrics_MultiCls) is unreachable from the bin/
command line -- train_bash.py runs with test_mode='val' and test_bash.py with 'test_only', and
the scorer is gated on `'test' in args.test_mode` -- so this kit does its own scoring.

Matching follows the CZ Imaging Institute convention: a prediction matches a ground-truth
particle of the same class when it lies within 0.5 x that particle's radius, one prediction per
particle, greedily by ascending distance (highest-score prediction first when scores exist).
Reported per class and macro-averaged: precision, recall, F1 and F-beta with beta=4 (the CZII
challenge metric, which weights recall heavily).

Both coordinate files use the same (cls, x, y, z) convention this kit writes in
prepare_from_datalist.py, so no re-mapping is needed:
  GT    <base>/raw_data/<tomo>.coords                          cls x y z
  pred  <run>/PredictedLabels/Coords_withArea/<tomo>.coords     cls x y z area
"""
import argparse
import json
import os

import numpy as np

from datalist_utils import read_coords

BETA = 4.0


def fbeta(p, r, beta=BETA):
    b2 = beta * beta
    return 0.0 if (p == 0 and r == 0) else (1 + b2) * p * r / (b2 * p + r + 1e-12)


def match_one_class(pred_xyz, pred_score, gt_xyz, tol):
    """Greedy one-to-one matching within `tol`. Returns (n_tp, [distances of the matches])."""
    if len(pred_xyz) == 0 or len(gt_xyz) == 0:
        return 0, []
    order = np.argsort(-pred_score) if pred_score is not None else np.arange(len(pred_xyz))
    taken = np.zeros(len(gt_xyz), dtype=bool)
    tp, dists = 0, []
    for i in order:
        d = np.linalg.norm(gt_xyz - pred_xyz[i], axis=1)
        d[taken] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            taken[j] = True
            tp += 1
            dists.append(float(d[j]))
    return tp, dists


def score_tomogram(gt, pred, radii, classes, tol_factor):
    """gt/pred: (N, >=4) arrays of (cls, x, y, z[, area]). Classes are 1-based in the files."""
    per_class, tp_all, fp_all, fn_all = {}, 0, 0, 0
    for c in classes:
        cls1 = c + 1                                  # files store cls+1
        g = gt[gt[:, 0] == cls1][:, 1:4]
        p = pred[pred[:, 0] == cls1]
        p_xyz = p[:, 1:4]
        p_score = p[:, 4] if p.shape[1] > 4 else None
        tol = tol_factor * radii[c]
        tp, dists = match_one_class(p_xyz, p_score, g, tol)
        fp, fn = len(p_xyz) - tp, len(g) - tp
        prec = tp / len(p_xyz) if len(p_xyz) else 0.0
        rec = tp / len(g) if len(g) else float("nan")
        per_class[str(c)] = {
            "n_gt": int(len(g)), "n_pred": int(len(p_xyz)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": prec, "recall": rec,
            "f1": fbeta(prec, rec, 1.0) if len(g) else float("nan"),
            f"f{BETA:g}": fbeta(prec, rec) if len(g) else float("nan"),
            "tolerance_voxels": tol,
            "avg_distance": float(np.mean(dists)) if dists else float("nan"),
        }
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn

    scored = [v for v in per_class.values() if v["n_gt"] > 0]
    micro_p = tp_all / (tp_all + fp_all) if (tp_all + fp_all) else 0.0
    micro_r = tp_all / (tp_all + fn_all) if (tp_all + fn_all) else 0.0
    return {
        "per_class": per_class,
        "n_gt": int(len(gt)), "n_pred": int(len(pred)),
        "tp": int(tp_all), "fp": int(fp_all), "fn": int(fn_all),
        "micro_precision": micro_p, "micro_recall": micro_r,
        "micro_f1": fbeta(micro_p, micro_r, 1.0),
        f"micro_f{BETA:g}": fbeta(micro_p, micro_r),
        "macro_precision": float(np.mean([v["precision"] for v in scored])) if scored else 0.0,
        "macro_recall": float(np.mean([v["recall"] for v in scored])) if scored else 0.0,
        "macro_f1": float(np.mean([v["f1"] for v in scored])) if scored else 0.0,
        f"macro_f{BETA:g}": float(np.mean([v[f"f{BETA:g}"] for v in scored])) if scored else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="DeepETPicker base_path (holds raw_data/)")
    ap.add_argument("--pred-dir", required=True,
                    help="the PredictedLabels dir test_bash.py wrote (or its Coords_withArea)")
    ap.add_argument("--tomos", default=None,
                    help="comma-separated tomograms to score (default: the test and val ids "
                         "recorded in dataset_split.json)")
    ap.add_argument("--tol-factor", type=float, default=0.5,
                    help="match radius as a fraction of the particle radius (CZII uses 0.5)")
    ap.add_argument("--out", default=None, help="metrics.json path (default: <pred-dir>/metrics.json)")
    args = ap.parse_args()

    with open(os.path.join(args.base, "dataset_split.json")) as f:
        split = json.load(f)
    names = split["tomograms"]
    radii = {int(k): float(v) for k, v in split["radii_voxels"].items()}
    classes = sorted(radii)

    if args.tomos:
        tomos = [t.strip() for t in args.tomos.split(",") if t.strip()]
    else:
        tomos = [names[i] for i in (split.get("test_id", len(names) - 1),
                                    split.get("val_id", len(names) - 2))]

    coords_dir = args.pred_dir
    if os.path.isdir(os.path.join(coords_dir, "Coords_withArea")):
        coords_dir = os.path.join(coords_dir, "Coords_withArea")

    per_image, roles = {}, {}
    for tomo in tomos:
        gt = read_coords(os.path.join(args.base, "raw_data", tomo + ".coords"))
        pred_file = os.path.join(coords_dir, tomo + ".coords")
        if not os.path.exists(pred_file):
            raise SystemExit(f"no prediction for {tomo}: {pred_file}")
        pred = read_coords(pred_file)
        per_image[tomo] = score_tomogram(gt, pred, radii, classes, args.tol_factor)
        i = names.index(tomo) if tomo in names else -1
        roles[tomo] = ("test" if i == split.get("test_id") else
                       "val" if i == split.get("val_id") else "train")

        m = per_image[tomo]
        print(f"\n=== {tomo}  ({roles[tomo]})  gt={m['n_gt']} pred={m['n_pred']}")
        print(f"    {'cls':>3} {'n_gt':>5} {'n_pred':>6} {'tp':>4} {'prec':>6} {'rec':>6} "
              f"{'f1':>6} {'f'+format(BETA,'g'):>6} {'tol':>5}")
        for c in classes:
            v = m["per_class"][str(c)]
            print(f"    {c:>3} {v['n_gt']:>5} {v['n_pred']:>6} {v['tp']:>4} "
                  f"{v['precision']:>6.3f} {v['recall']:>6.3f} {v['f1']:>6.3f} "
                  f"{v[f'f{BETA:g}']:>6.3f} {v['tolerance_voxels']:>5.1f}")
        print(f"    micro  P={m['micro_precision']:.3f} R={m['micro_recall']:.3f} "
              f"F1={m['micro_f1']:.3f} F{BETA:g}={m[f'micro_f{BETA:g}']:.3f}")
        print(f"    macro  P={m['macro_precision']:.3f} R={m['macro_recall']:.3f} "
              f"F1={m['macro_f1']:.3f} F{BETA:g}={m[f'macro_f{BETA:g}']:.3f}")

    out = args.out or os.path.join(args.pred_dir, "metrics.json")
    with open(out, "w") as f:
        json.dump({"per_image": per_image,
                   "roles": roles,
                   "config": {"method": "DeepETPicker",
                              "tol_factor": args.tol_factor,
                              "beta": BETA,
                              "radii_voxels": {str(c): radii[c] for c in classes},
                              "train_ids": split.get("train_ids"),
                              "val_id": split.get("val_id"),
                              "test_id": split.get("test_id"),
                              "tomograms": names}}, f, indent=2)
    print(f"\nmetrics -> {out}")


if __name__ == "__main__":
    main()
