#!/usr/bin/env python
"""Verify the multi-class wiring: is class i's probability map really channel i of the network?

Two checks, deliberately separated, because they answer different questions:

  A. EXTRACTION (training-independent, decisive).  Re-assemble every channel with DeePiCt's own
     assemble_tomo_from_subtomos and compare against what the pipeline wrote. If class i matches
     class_number=i, the per-class .predict_<class>.yaml mechanism is correct -- regardless of what
     the model learned. Matching is to float32 tolerance: the values round-trip through MRC, so the
     diagonal lands at ~1e-9, not exactly 0, while the off-diagonal sits at ~0.5-1.0.

  B. AGREEMENT WITH GROUND TRUTH (needs a trained model).  Dice of each class's map against each
     class's labels. An UNDERTRAINED model fails this without anything being wired wrong: channels
     collapse to a generic "foreground" detector, which best matches whichever class is largest.
     So a failure here is only meaningful if A passed AND the channels are well separated.

    python tests/check_class_mapping.py --out <run_dir> [--deepict <repo>]
"""
import argparse
import contextlib
import io
import itertools
import os
import shutil
import sys
import tempfile

import mrcfile
import numpy as np
import yaml

FLOAT32_TOL = 1e-6      # MRC round-trip noise; the off-diagonal is ~0.5-1.0, so this is ample


def rd(p):
    with mrcfile.open(p, permissive=True) as m:
        return np.asarray(m.data).astype(np.float32)


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a * b).sum() / s) if s else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run dir holding config.yaml")
    ap.add_argument("--deepict", default="/scratch/cyyu/cryodino/deepict/DeePiCt")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(args.out, "config.yaml")))
    classes = cfg["training"]["semantic_classes"]
    if len(classes) < 2:
        print(f"only {len(classes)} class ({classes}) -- nothing to mismap, skipping")
        return 0
    model = os.path.basename(cfg["model_path"])[:-4]
    preds = os.path.join(cfg["output_dir"], "predictions", model)
    masks = os.path.join(args.out, "data", "masks")
    box, ov = cfg["training"]["box_size"], cfg["training"]["overlap"]

    sys.path.insert(0, os.path.join(args.deepict, "3d_cnn", "src"))
    from file_actions.writers.h5 import assemble_tomo_from_subtomos

    fails, warns = [], []
    for tomo in cfg["tomos_sets"]["prediction_list"]:
        print(f"\n=== {tomo} ===")
        got = {c: rd(os.path.join(preds, tomo, c, "probability_map.mrc")) for c in classes}
        shape = got[classes[0]].shape
        part = os.path.join(args.out, "work", "testing_data", tomo, "partition.h5")

        # ---- A. extraction
        tmp = tempfile.mkdtemp()
        direct = []
        for n in range(len(classes)):
            p = os.path.join(tmp, f"ch{n}.mrc")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                assemble_tomo_from_subtomos(
                    output_path=p, partition_file_path=part, output_shape=shape,
                    subtomo_shape=(box, box, box),
                    subtomos_internal_path=f"volumes/predictions/{model}",
                    class_number=n, overlap=ov, reconstruction_type="prediction",
                    final_activation="sigmoid")
            direct.append(rd(p))
        print("  A. extraction: max|pipeline - assembler(class_number=N)|")
        print("     " + " " * 8 + "".join(f"{'cn='+str(j):>13s}" for j in range(len(classes))))
        for i, c in enumerate(classes):
            row = [float(np.abs(got[c] - direct[j]).max()) for j in range(len(classes))]
            ok = row[i] < FLOAT32_TOL and min(range(len(row)), key=lambda k: row[k]) == i
            print(f"     {c:>8s}" + "".join(f"{v:13.2e}" for v in row) +
                  f"   {'ok' if ok else 'MISMAPPED'}")
            if not ok:
                fails.append(f"{tomo}/{c}: assembled from the wrong channel")
        shutil.rmtree(tmp, ignore_errors=True)

        # ---- channel separation, needed to interpret B
        print("  channel separation (correlation between predicted classes):")
        collapsed = []
        for a, b in itertools.combinations(classes, 2):
            r = float(np.corrcoef(got[a].ravel(), got[b].ravel())[0, 1])
            note = "  <- nearly redundant" if r > 0.95 else ""
            print(f"     corr({a},{b}) = {r:+.4f}{note}")
            if r > 0.95:
                collapsed.append(f"{a}~{b}")

        # ---- B. agreement with ground truth
        gt = {c: rd(os.path.join(masks, f"{tomo}_{c}.mrc")) for c in classes}
        sl = tuple(slice(0, s) for s in np.min([list(v.shape) for v in
                                                list(got.values()) + list(gt.values())], axis=0))
        print(f"  B. Dice(pred>{args.threshold}, GT)   rows=predicted, cols=ground truth")
        print("     " + " " * 8 + "".join(f"{c:>13s}" for c in classes))
        diag_ok = True
        for i, c in enumerate(classes):
            b = (got[c][sl] > args.threshold).astype(np.float32)
            row = [dice(b, gt[g][sl]) for g in classes]
            best = int(np.nanargmax(row)) if np.nansum(row) > 0 else -1
            hit = best == i
            diag_ok &= hit
            print(f"     {c:>8s}" + "".join(f"{v:13.4f}" for v in row) +
                  f"   {'ok' if hit else ('empty' if best < 0 else 'best=' + classes[best])}")
        if not diag_ok:
            # A failure here is about the MODEL, not the wiring: check A already proved class i is
            # assembled from channel i, and the masks are verified per class at prepare time. The
            # usual causes are undertraining (channels still redundant) or a rare class that does
            # not generalise. To tell them apart, predict on a TRAINING tomogram -- if the diagonal
            # is clean there, it is generalisation.
            warns.append(
                f"{tomo}: ground-truth diagonal does not dominate. Extraction is proven correct "
                f"(check A), so this is model quality, not wiring"
                + (f"; channels {', '.join(collapsed)} are still redundant -> undertrained"
                   if collapsed else
                   "; channels are well separated -> likely a rare class failing to generalise. "
                   "Add a training tomogram to prediction_list: a clean diagonal there confirms it"))

    print()
    for w in warns:
        print("WARNING:", w)
    print("\n" + ("WIRING OK" + (" -- extraction proven; ground-truth agreement is limited by model "
          "quality, see warnings above" if warns else " -- extraction proven and the GT diagonal dominates")
          if not fails else "WIRING FAILURES:\n  " + "\n  ".join(fails)))
    # exit status reflects the WIRING (check A), which is what this kit is responsible for;
    # model quality is reported as warnings.
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
