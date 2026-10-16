#!/usr/bin/env python
"""Known-answer test for evaluate.py -- no GPU, no trained model needed.

Builds a synthetic prediction file from a tomogram's real ground truth: keeps a known fraction of
the particles (jittered by less than the match tolerance), drops the rest, and adds a known number
of false positives far from anything.  evaluate.py must recover exactly those tp/fp/fn counts and
the closed-form precision/recall/F-beta.  Also checks that a pick placed just *outside* the
tolerance does not match.

    python tests/test_evaluate.py --base <out_dir> [--tomo TS_99_9]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BETA = 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="a run's out_dir (needs raw_data/ + split json)")
    ap.add_argument("--tomo", default=None, help="default: the split's test tomogram")
    ap.add_argument("--keep", type=float, default=0.7, help="fraction of GT the fake picker finds")
    ap.add_argument("--n-fp", type=int, default=17, help="false positives injected per class")
    ap.add_argument("--workdir", default=None, help="default: a temp dir, removed afterwards")
    args = ap.parse_args()

    split = json.load(open(os.path.join(args.base, "dataset_split.json")))
    tomo = args.tomo or split["tomograms"][split.get("test_id", -1)]
    radii = {int(k): float(v) for k, v in split["radii_voxels"].items()}
    rng = np.random.default_rng(0)

    tmp = args.workdir or tempfile.mkdtemp(prefix="dep_eval_test_")
    try:
        os.makedirs(f"{tmp}/raw_data", exist_ok=True)
        os.makedirs(f"{tmp}/pred/Coords_withArea", exist_ok=True)
        shutil.copy(os.path.join(args.base, "raw_data", f"{tomo}.coords"),
                    f"{tmp}/raw_data/{tomo}.coords")
        json.dump(split, open(f"{tmp}/dataset_split.json", "w"))

        gt = np.loadtxt(f"{tmp}/raw_data/{tomo}.coords", delimiter="\t", ndmin=2)
        shape = split["volume_shape_xyz"][tomo]

        rows, expect = [], {}
        for c in sorted(radii):
            g = gt[gt[:, 0] == c + 1]
            tol = 0.5 * radii[c]
            n_keep = int(round(len(g) * args.keep))
            for i in rng.choice(len(g), n_keep, replace=False):     # true positives
                jit = rng.normal(0, tol / 6, 3).clip(-tol * 0.5, tol * 0.5)
                rows.append([c + 1, *(g[i, 1:4] + jit), rng.uniform(1, 100)])
            for _ in range(args.n_fp):                              # false positives, far away
                while True:
                    p = np.array([rng.integers(0, s) for s in shape])
                    if np.linalg.norm(gt[:, 1:4] - p, axis=1).min() > 60:
                        break
                rows.append([c + 1, *p, rng.uniform(1, 100)])
            expect[c] = dict(n_gt=len(g), tp=n_keep, fp=args.n_fp, fn=len(g) - n_keep)

        np.savetxt(f"{tmp}/pred/Coords_withArea/{tomo}.coords", np.array(rows),
                   delimiter="\t", fmt="%.4f")

        r = subprocess.run([sys.executable, f"{KIT}/evaluate.py", "--base", tmp,
                            "--pred-dir", f"{tmp}/pred", "--tomos", tomo],
                           cwd=KIT, capture_output=True, text=True)
        print(r.stdout)
        if r.returncode:
            print(r.stderr)
            return 1

        got = json.load(open(f"{tmp}/pred/metrics.json"))["per_image"][tomo]
        fails = []
        for c, e in expect.items():
            v = got["per_class"][str(c)]
            for k in ("n_gt", "tp", "fp", "fn"):
                if v[k] != e[k]:
                    fails.append(f"class {c} {k}: got {v[k]}, injected {e[k]}")
            p, rc = e["tp"] / (e["tp"] + e["fp"]), e["tp"] / e["n_gt"]
            b2 = BETA ** 2
            fb = (1 + b2) * p * rc / (b2 * p + rc)
            if abs(v["precision"] - p) > 1e-9 or abs(v["recall"] - rc) > 1e-9:
                fails.append(f"class {c} P/R: got {v['precision']:.4f}/{v['recall']:.4f}, "
                             f"expected {p:.4f}/{rc:.4f}")
            if abs(v[f"f{BETA:g}"] - fb) > 1e-9:
                fails.append(f"class {c} f{BETA:g}: got {v[f'f{BETA:g}']:.4f}, expected {fb:.4f}")
        for k in ("tp", "fp", "fn"):
            tot = sum(e[k] for e in expect.values())
            if got[k] != tot:
                fails.append(f"total {k}: got {got[k]}, injected {tot}")

        # a pick just beyond 0.5*radius must NOT be counted as a hit
        c = max(radii, key=radii.get)
        one = gt[gt[:, 0] == c + 1][0]
        far = np.array([[c + 1, one[1] + 0.5 * radii[c] + 3, one[2], one[3], 50.0]])
        np.savetxt(f"{tmp}/pred/Coords_withArea/{tomo}.coords", far, delimiter="\t", fmt="%.4f")
        subprocess.run([sys.executable, f"{KIT}/evaluate.py", "--base", tmp, "--pred-dir",
                        f"{tmp}/pred", "--tomos", tomo], cwd=KIT, capture_output=True, text=True)
        beyond = json.load(open(f"{tmp}/pred/metrics.json"))["per_image"][tomo]["per_class"][str(c)]
        if beyond["tp"] != 0:
            fails.append(f"a pick beyond 0.5*radius matched anyway (tp={beyond['tp']})")

        print("ALL PASS" if not fails else "FAILURES:\n  " + "\n  ".join(fails))
        return 1 if fails else 0
    finally:
        if not args.workdir:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
