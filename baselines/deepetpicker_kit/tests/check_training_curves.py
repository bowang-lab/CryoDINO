#!/usr/bin/env python
"""Judge a run's learning curves: is train/val loss falling and val performance rising?

    python tests/check_training_curves.py --base <out_dir> [--run-dir <version_N>]

Reports **INCONCLUSIVE** (exit 2) rather than a pass when there are too few epochs to have a
trend -- a 1-epoch run logs one point per metric, which says nothing about whether the model
learns.  Exit 0 when the curves are healthy, 1 when a metric moves the wrong way.
"""
import argparse
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOSSES = ["train_loss", "val_loss"]
PERF = ["val_f1", "val_precision", "val_recall", "val_iou"]
MIN_EPOCHS = 5


def trend(v):
    """(first, last, delta) using end-block means, to damp per-epoch noise."""
    k = max(1, len(v) // 5)
    a, b = sum(v[:k]) / k, sum(v[-k:]) / k
    return a, b, b - a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="a run's out_dir; picks the newest version_*")
    ap.add_argument("--run-dir", default=None, help="a specific version_N directory")
    args = ap.parse_args()
    if not (args.base or args.run_dir):
        ap.error("need --base or --run-dir")

    run = args.run_dir
    if run is None:
        cands = glob.glob(os.path.join(args.base, "runs", "*", "*", "version_*"))
        if not cands:
            print(f"no runs under {args.base}/runs")
            return 1
        run = max(cands, key=os.path.getmtime)
    print(f"run: {run}\n")

    ea = EventAccumulator(run, size_guidance={"scalars": 0})
    ea.Reload()
    series = {t: [e.value for e in ea.Scalars(t)] for t in ea.Tags()["scalars"]}
    n = len(series.get("val_loss", []))
    print(f"epochs logged: {n}")
    if n < MIN_EPOCHS:
        print(f"\nINCONCLUSIVE -- {n} epoch(s) is not enough to judge a trend "
              f"(need >= {MIN_EPOCHS}). A single point has no slope.")
        return 2

    bad = []
    print(f"\n{'metric':16s} {'first':>9s} {'last':>9s} {'delta':>9s}  want  result")
    for t in LOSSES + PERF:
        v = series.get(t)
        if not v:
            continue
        a, b, d = trend(v)
        want_down = t in LOSSES
        good = (d < 0) if want_down else (d > 0)
        if not good:
            bad.append(t)
        print(f"{t:16s} {a:9.4f} {b:9.4f} {d:+9.4f}  {'down' if want_down else 'up':4s}  "
              f"{'ok' if good else 'WRONG WAY'}")

    print()
    for t in LOSSES + PERF:
        if series.get(t):
            print(f"{t:16s} " + " ".join(f"{x:.4f}" for x in series[t]))

    vl = series.get("val_loss", [])
    if vl:
        i = min(range(len(vl)), key=lambda j: vl[j])
        tail = "still improving at the end" if i >= len(vl) - 2 else \
               "val_loss bottomed out early -- later epochs overfit (the checkpoint " \
               "monitors val_loss, so the saved model is the best one, not the last)"
        print(f"\nbest val_loss {vl[i]:.4f} at epoch {i} of {len(vl) - 1}  -- {tail}")

    print("\n" + ("CURVES OK -- loss falling, val performance rising" if not bad else
                  "CURVES PROBLEM -- wrong direction: " + ", ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
