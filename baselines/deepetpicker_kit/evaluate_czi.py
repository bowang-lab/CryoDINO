#!/usr/bin/env python
"""Score DeepETPicker predictions on the datalist's validation/test splits with CryoDINO's own
detection metrics (3DINO/dinov2/eval/detection_3d/metrics.py), so the numbers are directly
comparable with what detection3d.py reports for CryoDINO / ResNetFPN.

  --dataset czi (default): CZIDetectionMetrics -- 6 classes, weighted F4 (beta-amylase weight 0),
                           per-class score thresholds.  Output keys val_f4 / test_f4 / ...
  --dataset byu:           BYUDetectionMetrics  -- 1 class (motor), F-beta (beta from the class,
                           2), match radius min_radius / voxel_size per tomogram, one global score
                           threshold.  Output keys val_f2 / test_f2 / ...

Protocol, identical to detection3d.py:
  * ground truth = the datalist `points` of each entry (x, y, z, class_id, radius), -100 rows
    dropped by the metric's accumulate(); voxel_size from the entry (BYU: varies per tomogram);
  * predictions  = <pred-root>/<split>/<name>.coords from infer_splits.py
    (cls+1, x, y, z, score), x/y/z in the same datalist voxel convention;
  * one metric object per split, accumulated over every tomogram, then threshold_sweep().

Two test numbers are written: test_<score>_at_val_thresholds applies the threshold(s) chosen on
validation -- what detection3d.py reports since upstream bfbb248 (no sweep on test), so this is the
number to compare -- and test_<score>, a sweep on the test set itself (optimistic; kept for
reference, it is what detection3d.py reported before that commit).

BYU note: the match radius is BYUDetectionMetrics' default (1000 A since bfbb248, the official BYU
2025 value; it was 500 A before) and every prediction is scored, while the official BYU metric
keeps one prediction per tomogram. --byu-min-radius overrides the radius (same class otherwise).

The metric module is loaded straight from its file (not `import dinov2...`), so this needs only
numpy + scipy -- it runs in the DeepETPicker venv.
"""
import argparse
import importlib.util
import json
import os

import numpy as np

from datalist_utils import load_datalist, read_coords, strip_ext, valid_points

KIT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METRICS = os.path.join(KIT, "..", "..", "3DINO", "dinov2", "eval", "detection_3d",
                               "metrics.py")
SHORT = {"validation": "val", "test": "test"}


def load_metrics_module(path):
    spec = importlib.util.spec_from_file_location("cryodino_detection_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Scorer:
    """Thin adapter so CZI (6 classes, per-class thresholds) and BYU (1 class, one threshold)
    produce the same result layout."""

    def __init__(self, dataset, mod, byu_min_radius=None):
        self.dataset = dataset
        if dataset == "czi":
            self.cls = mod.CZIDetectionMetrics
            self.names = list(self.cls.PARTICLE_NAMES)
            self.score_key = "f%d" % mod.CZIDetectionMetrics().beta
            self.new = self.cls
        else:
            self.cls = mod.BYUDetectionMetrics
            self.names = list(self.cls.PARTICLE_NAMES)        # ['motor']
            kw = {} if byu_min_radius is None else {"min_radius": byu_min_radius}
            self.new = lambda: self.cls(**kw)
            probe = self.new()
            self.score_key = "f%g" % probe.beta
            self.min_radius = probe.min_radius

    def accumulate(self, metric, pred, labels, scores, gt_all, vs):
        if self.dataset == "czi":
            metric.accumulate(pred[:, 1:4], labels, scores, gt_all, vs)
        else:
            metric.accumulate(pred[:, 1:4], scores, gt_all, vs)

    def sweep(self, metric):
        """-> (score, {name: threshold}, {name: per-class score})"""
        s, thr, per_cls = metric.threshold_sweep()
        if not isinstance(thr, dict):                      # BYU: one global threshold
            thr = {n: float(thr) for n in self.names}
        return float(s), {k: float(v) for k, v in thr.items()}, {k: float(v) for k, v in per_cls.items()}

    def at_thresholds(self, metric, thresholds):
        """Score with fixed threshold(s), e.g. the ones chosen on validation."""
        if self.dataset == "byu":
            s, per_cls = metric._fbeta_at_threshold(float(thresholds[self.names[0]]))
            return float(s), {k: float(v) for k, v in per_cls.items()}
        per_cls = {}
        for name, thr in thresholds.items():
            _, pc = metric._f4_at_threshold(float(thr))
            per_cls[name] = float(pc[name])
        w = dict(zip(metric.PARTICLE_NAMES, metric.WEIGHTS))
        return sum(per_cls[n] * w[n] for n in per_cls) / sum(w.values()), per_cls


def check_class_radii(gt, voxel_size, metric_cls, name, tol_ang):
    """CZI only: datalist radius x voxel_size must equal the metric's radius for that class id.
    Catches a class-order mismatch between the datalist and CZIDetectionMetrics."""
    radii = metric_cls.PARTICLE_RADII_ANG
    for row in gt:
        c = int(row[3])
        if not 0 <= c < len(radii):
            raise SystemExit(f"{name}: GT class id {c} outside 0..{len(radii) - 1}")
        if abs(row[4] * voxel_size - radii[c]) > tol_ang:
            raise SystemExit(f"{name}: class {c} radius {row[4]:.3f} vox x {voxel_size} = "
                             f"{row[4] * voxel_size:.1f} A, metric expects {radii[c]} A "
                             f"({metric_cls.PARTICLE_NAMES[c]}) -- class order or units differ "
                             f"(--skip-radius-check to override)")


def score_split(entries, pred_dir, sc, args):
    metric = sc.new()
    names = sc.names
    n_gt = np.zeros(len(names), dtype=int)
    n_pred = np.zeros(len(names), dtype=int)
    missing, empty_gt = [], []
    for e in entries:
        name = strip_ext(e["image"])
        gt_all = np.asarray(e.get("points", []), dtype=np.float32).reshape(-1, 5)
        gt = valid_points(gt_all)
        if len(gt) == 0:
            empty_gt.append(name)
        if "voxel_size" in e:
            vs = float(e["voxel_size"])
        elif args.voxel_size is not None:
            vs = args.voxel_size
        else:
            raise SystemExit(f"{name}: no voxel_size in the datalist entry; pass --voxel-size")
        if sc.dataset == "czi" and not args.skip_radius_check:
            check_class_radii(gt, vs, sc.cls, name, args.radius_tol)
        if len(gt) and (gt[:, 3].min() < 0 or gt[:, 3].max() >= len(names)):
            raise SystemExit(f"{name}: GT class ids {sorted(set(gt[:, 3].astype(int)))} not in "
                             f"0..{len(names) - 1} for --dataset {sc.dataset}")

        pf = os.path.join(pred_dir, name + ".coords")
        if not os.path.exists(pf):
            missing.append(name)
            if args.allow_missing:
                continue
            raise SystemExit(f"no prediction for {name}: {pf} (run infer_splits.py, or "
                             f"--allow-missing to score what exists)")
        pred = read_coords(pf)
        labels = pred[:, 0].astype(int) - 1                 # .coords classes are 1-based
        if len(labels) and (labels.min() < 0 or labels.max() >= len(names)):
            raise SystemExit(f"{name}: predicted class ids {sorted(set(labels + 1))} not in "
                             f"1..{len(names)}")
        scores = pred[:, 4] if pred.shape[1] > 4 else np.ones(len(pred))
        sc.accumulate(metric, pred, labels, scores, gt_all, vs)
        n_gt += np.bincount(gt[:, 3].astype(int), minlength=len(names))
        n_pred += np.bincount(labels, minlength=len(names))

    n_scored = len(entries) - (len(missing) if args.allow_missing else 0)
    if n_scored == 0:
        raise SystemExit("nothing to score")
    if len(empty_gt) == len(entries):
        raise SystemExit("every tomogram has zero labelled points (all -100) -- this datalist's "
                         "split carries no ground truth")
    s, thr, per_cls = sc.sweep(metric)
    k = sc.score_key
    return metric, {
        k: s,
        f"per_cls_{k}": per_cls,
        "thresholds": thr,
        "n_tomograms": n_scored,
        "n_missing": len(missing),
        "n_tomograms_without_gt": len(empty_gt),
        "n_gt_per_cls": {c: int(v) for c, v in zip(names, n_gt)},
        "n_pred_per_cls": {c: int(v) for c, v in zip(names, n_pred)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", required=True)
    ap.add_argument("--pred-root", required=True, help="the checkpoint's predictions folder (holds <split>/)")
    ap.add_argument("--dataset", default="czi", choices=["czi", "byu"])
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--limit", type=int, default=None, help="only the first N of each split")
    ap.add_argument("--metrics-py", default=DEFAULT_METRICS,
                    help="path to 3DINO/dinov2/eval/detection_3d/metrics.py")
    ap.add_argument("--voxel-size", type=float, default=None,
                    help="fallback A/voxel for entries without voxel_size")
    ap.add_argument("--byu-min-radius", type=float, default=None,
                    help="BYU match radius in A (default: BYUDetectionMetrics' own, 1000 since "
                         "upstream bfbb248; it was 500 before)")
    ap.add_argument("--radius-tol", type=float, default=2.0, help="A, for the CZI class-order check")
    ap.add_argument("--skip-radius-check", action="store_true")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--out", default=None, help="default: <pred-root>/metrics_<dataset>.json "
                                                "(czi keeps the name metrics_czi.json)")
    args = ap.parse_args()

    mod = load_metrics_module(os.path.abspath(args.metrics_py))
    sc = Scorer(args.dataset, mod, args.byu_min_radius)
    k = sc.score_key
    d = load_datalist(args.datalist)
    results, metrics = {}, {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        entries = d[split][: args.limit] if args.limit else d[split]
        metrics[split], r = score_split(entries, os.path.join(args.pred_root, split), sc, args)
        key = SHORT.get(split, split)
        results.update({f"{key}_{n}": v for n, v in r.items()})

    if "validation" in metrics and "test" in metrics:
        s, pc = sc.at_thresholds(metrics["test"], results["val_thresholds"])
        results[f"test_{k}_at_val_thresholds"] = s
        results[f"test_per_cls_{k}_at_val_thresholds"] = pc

    print()
    cols = [c for c in ("val", "test") if f"{c}_{k}" in results]
    weights = dict(zip(sc.names, getattr(sc.cls, "WEIGHTS", [1] * len(sc.names))))
    K = k.upper()
    print(f"{'class':22s} {'weight':>6} " + " ".join(f"{c + ' ' + K:>9} {c + ' thr':>8}" for c in cols)
          + (f"  test {K}@val-thr" if f"test_{k}_at_val_thresholds" in results else ""))
    for n in sc.names:
        row = f"{n:22s} {weights[n]:>6} " + " ".join(
            f"{results[c + '_per_cls_' + k][n]:>9.4f} {results[c + '_thresholds'][n]:>8.4f}" for c in cols)
        if f"test_{k}_at_val_thresholds" in results:
            row += f"  {results['test_per_cls_' + k + '_at_val_thresholds'][n]:>15.4f}"
        print(row)
    for c in cols:
        print(f"{c} {K} = {results[c + '_' + k]:.4f}   over {results[c + '_n_tomograms']} tomograms, "
              f"gt={sum(results[c + '_n_gt_per_cls'].values())} "
              f"pred={sum(results[c + '_n_pred_per_cls'].values())}")
    if f"test_{k}_at_val_thresholds" in results:
        print(f"test {K} at val-chosen thresholds = {results[f'test_{k}_at_val_thresholds']:.4f}")

    cfg = {"method": "DeepETPicker", "dataset": args.dataset, "score": k,
           "metric": sc.cls.__name__, "metrics_py": os.path.abspath(args.metrics_py),
           "datalist": os.path.abspath(args.datalist),
           "score_column": "Coords_withArea col 5 (test.py _nms_v2 h_val)"}
    if args.dataset == "byu":
        cfg["byu_min_radius_A"] = sc.min_radius
    results["config"] = cfg
    out = args.out or os.path.join(args.pred_root, f"metrics_{args.dataset}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nmetrics -> {out}")


if __name__ == "__main__":
    main()
