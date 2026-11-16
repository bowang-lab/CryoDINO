#!/usr/bin/env python
"""Pick the periodic checkpoint with the highest validation F4 (CZIDetectionMetrics), the way
detection3d.py picks CryoDINO's best_model.pth.

Reads <out>/periodic_checkpoints.txt (written by run_benchmark.sh when SAVE_EVERY > 0) and each
checkpoint's <out>/predictions/<version>_<name>/metrics_<czi|byu>.json (written by its validation
scoring job). CZI selects on val F4, BYU on val F2 (whatever score evaluate_czi.py recorded).
Writes
  <out>/checkpoint_selection.csv   epoch, val score, per-class val score, checkpoint  (the curve)
  <out>/best_f4_checkpoint.txt     the winner (ties -> earliest epoch)
"""
import argparse
import csv
import json
import os
import re

from infer_splits import ckpt_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the run's base_path")
    args = ap.parse_args()
    out = os.path.abspath(args.out)

    with open(os.path.join(out, "periodic_checkpoints.txt")) as f:
        ckpts = [l.strip() for l in f if l.strip()]
    if not ckpts:
        raise SystemExit("periodic_checkpoints.txt is empty")

    # opt-in: SKIP_CKPT_EPOCHS="4" (comma-separated) leaves those periodic checkpoints out of the
    # selection, e.g. a barely trained one whose validation scoring cannot finish; they are listed
    # in <out>/skipped_checkpoints.txt and in the printout. Unset = every checkpoint must be scored.
    skip = {int(e) for e in os.environ.get("SKIP_CKPT_EPOCHS", "").replace(" ", "").split(",") if e}
    skipped = [c for c in ckpts
               if re.search(r"epoch=(\d+)", os.path.basename(c))
               and int(re.search(r"epoch=(\d+)", os.path.basename(c)).group(1)) in skip]
    if skip and len(skipped) != len(skip):
        raise SystemExit(f"SKIP_CKPT_EPOCHS={sorted(skip)} but only matched {len(skipped)} checkpoint(s)")
    ckpts = [c for c in ckpts if c not in skipped]
    with open(os.path.join(out, "skipped_checkpoints.txt"), "w") as f:
        f.writelines(c + "\n" for c in skipped)

    rows, missing = [], []
    for c in ckpts:
        d = os.path.join(out, "predictions", ckpt_tag(c))
        found = [os.path.join(d, f) for f in ("metrics_czi.json", "metrics_byu.json")
                 if os.path.exists(os.path.join(d, f))]
        if not found:
            missing.append(d)
            continue
        r = json.load(open(found[0]))
        k = r.get("config", {}).get("score", "f4")
        if f"val_{k}" not in r:
            raise SystemExit(f"{found[0]} has no validation score")
        ep = re.search(r"epoch=(\d+)", os.path.basename(c))
        rows.append({"epoch": int(ep.group(1)) if ep else -1, "val_score": r[f"val_{k}"],
                     "score": k, "n_val_tomograms": r["val_n_tomograms"], **r[f"val_per_cls_{k}"],
                     "checkpoint": c})
    if missing:
        raise SystemExit(f"{len(missing)}/{len(ckpts)} checkpoints have no validation score, e.g. "
                         f"{missing[0]} -- rerun their scoring jobs first")
    n_val = {r["n_val_tomograms"] for r in rows}
    if len(n_val) != 1:
        raise SystemExit(f"checkpoints were scored on different numbers of tomograms: {n_val}")

    rows.sort(key=lambda r: r["epoch"])
    best = max(rows, key=lambda r: (r["val_score"], -r["epoch"]))
    keys = list(rows[0])
    with open(os.path.join(out, "checkpoint_selection.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "best_f4_checkpoint.txt"), "w") as f:
        f.write(best["checkpoint"] + "\n")

    K = rows[0]["score"].upper()
    print(f"validation {K} per checkpoint ({n_val.pop()} validation tomograms each):")
    for r in rows:
        print(f"  epoch {r['epoch']:3d}  val {K} {r['val_score']:.4f}" + ("   <- best" if r is best else ""))
    for c in skipped:
        print(f"  SKIPPED (SKIP_CKPT_EPOCHS, not scored): {c}")
    print(f"\nbest: epoch {best['epoch']}  val {K} {best['val_score']:.4f}  {best['checkpoint']}")
    print(f"-> {os.path.join(out, 'checkpoint_selection.csv')}, "
          f"{os.path.join(out, 'best_f4_checkpoint.txt')}")


if __name__ == "__main__":
    main()
