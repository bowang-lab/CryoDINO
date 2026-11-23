"""Check saved test predictions reproduce each run's results.json test F4 (CZII).

For every run dir given:
  0. If <run>/test_eval/test_predictions.pkl is the early format (no CSV / no format_version), convert it
     in place to format_version 1 (+ test_predictions.csv) via prediction_io.save_test_predictions;
     the original is kept as test_predictions_v0.pkl.
  A. pkl path: load metric._data back into CZIDetectionMetrics and call f4_at_thresholds(val thresholds)
     — the exact function detection3d.py uses for test.
  B. CSV path (independent): read test_predictions.csv rows with above_val_threshold == 1, take GT from the
     datalist (not from the pkl), count TP/FP/FN per tomogram/class with compute_metrics, pool, weighted F4.
  Pass requires (exact, 1e-9): A == test_f4 stored by the inference job; B == A (all per-class F4 too);
  every test tomogram present; GT in pkl == datalist GT; CSV rows == pkl candidates.
  Versus results.json (the original training-time test): |diff| <= 1e-3 overall and per class. A re-run
  on another GPU / torch build can differ by a detection or two (floating point), so this is reported,
  not required to be exact.
Usage: python verify_test_predictions.py --datalist <czi_100_datalist.json> <run_dir> [<run_dir> ...]
       (the test split is identical in the 100/50/10% datalists)
"""
import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "3DINO"))
from dinov2.eval.detection_3d.metrics import CZIDetectionMetrics, compute_metrics  # noqa: E402
from dinov2.eval.detection_3d.prediction_io import save_test_predictions  # noqa: E402



def stem(p):
    return os.path.basename(p).replace(".nii.gz", "")


def weighted_f4(tot, names, weights, beta=4):
    per = {}
    for c, n in enumerate(names):
        tp, fp, fn = tot[c]
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        per[n] = (1 + beta**2) * p * r / (beta**2 * p + r) if p + r > 0 else 0.0
    return sum(per[n] * w for n, w in zip(names, weights)) / sum(weights), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", required=True)
    ap.add_argument("--subdir", default="test_eval")
    ap.add_argument("--resave", action="store_true", help="rewrite pkl/csv from the pkl data (current format)")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()

    with open(a.datalist) as f:
        test = json.load(f)["test"]
    gt_by = {stem(e["image"]): (np.array(e["points"], float).reshape(-1, 5), float(e["voxel_size"])) for e in test}
    names, radii, weights = CZIDetectionMetrics.PARTICLE_NAMES, CZIDetectionMetrics.PARTICLE_RADII_ANG, CZIDetectionMetrics.WEIGHTS

    all_ok = True
    for run in a.runs:
        d = os.path.join(run, a.subdir)
        pkl = os.path.join(d, "test_predictions.pkl")
        res = json.load(open(os.path.join(run, "results.json")))
        ref, ref_cls = res["test_f4"], res["test_per_cls_f4"]
        msgs, ok = [], True
        if not os.path.exists(pkl):
            print(f"MISSING {pkl}"); all_ok = False; continue
        P = pickle.load(open(pkl, "rb"))

        # 0. convert early format
        if a.resave or P.get("format_version") != 1 or not os.path.exists(os.path.join(d, "test_predictions.csv")):
            if P.get("format_version") != 1:
                os.replace(pkl, os.path.join(d, "test_predictions_v0.pkl"))
            m = CZIDetectionMetrics(); m._data = P["data"]
            extra = {k: v for k, v in P.items() if k not in ("images", "data", "val_thresholds")}
            extra.setdefault("checkpoint", os.path.join(run, "best_model.pth"))
            save_test_predictions(d, P["images"], [gt_by[stem(i)][1] for i in P["images"]], m, P["val_thresholds"], extra)
            P = pickle.load(open(pkl, "rb"))
            msgs.append("re-saved (v1)")
        thr = P["val_thresholds"]
        if thr != res["best_val_thresholds"]:
            ok = False; msgs.append("val thresholds differ from results.json")

        # coverage + GT identity
        imgs = [stem(i) for i in P["images"]]
        if sorted(imgs) != sorted(gt_by):
            ok = False; msgs.append(f"tomograms: {len(imgs)} in pkl vs {len(gt_by)} in datalist")
        for i, dd in zip(imgs, P["data"]):
            g, vs = gt_by[i]
            if len(g) != len(dd["gt_labels"]) or not np.allclose(g[:, :3] * vs, dd["gt_centers_ang"], atol=1e-3) \
                    or not np.array_equal(g[:, 3].astype(int), dd["gt_labels"]):
                ok = False; msgs.append(f"GT mismatch {i}"); break

        # A. pkl -> metric code
        m = CZIDetectionMetrics(); m._data = P["data"]
        fa, fa_cls = m.f4_at_thresholds(thr)

        # B. CSV + datalist GT, own pooling
        preds = defaultdict(list)
        n_rows = 0
        with open(os.path.join(d, "test_predictions.csv")) as f:
            for r in csv.DictReader(f):
                n_rows += 1
                if r["above_val_threshold"] == "1":
                    preds[r["tomogram"]].append((float(r["x_ang"]), float(r["y_ang"]), float(r["z_ang"]), int(r["class_id"])))
        tot = np.zeros((len(names), 3))
        for t, (g, vs) in gt_by.items():
            p = np.array(preds.get(t, []), float).reshape(-1, 4)
            for c in range(len(names)):
                tot[c] += compute_metrics(g[g[:, 3] == c, :3] * vs, radii[c] * 0.5, p[p[:, 3] == c, :3])
        fb, fb_cls = weighted_f4(tot, names, weights)
        n_pkl = sum(len(dd["pred_scores"]) for dd in P["data"])
        if n_rows != n_pkl:
            ok = False; msgs.append(f"CSV rows {n_rows} != pkl candidates {n_pkl}")

        EX = 1e-9
        job_f4 = P.get("test_f4")
        if job_f4 is None or abs(fa - job_f4) > EX:
            ok = False; msgs.append(f"A pkl F4 {fa:.9f} != inference-job F4 {job_f4}")
        if abs(fb - fa) > EX or any(abs(fb_cls[n] - fa_cls[n]) > EX for n in names):
            ok = False; msgs.append(f"B csv F4 {fb:.9f} != A {fa:.9f}")
        d_ref = max([abs(fa - ref)] + [abs(fa_cls[n] - ref_cls[n]) for n in names])
        if d_ref > 1e-3:
            ok = False; msgs.append(f"differs from results.json by {d_ref:.2e}")
        msgs.append("identical to results.json" if d_ref <= EX else f"max |diff| vs results.json {d_ref:.1e}")
        all_ok &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {os.path.basename(run.rstrip('/'))}: results.json {ref:.6f} | "
              f"pkl {fa:.6f} | csv {fb:.6f} | {len(imgs)} tomos, {n_pkl} candidates"
              + (f"  [{'; '.join(msgs)}]" if msgs else ""), flush=True)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
