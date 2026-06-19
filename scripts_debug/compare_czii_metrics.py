#!/usr/bin/env python
"""
Verify that the CZII F-beta detection metric implemented in

    3DINO/dinov2/eval/detection_3d/metrics.py            (CZIDetectionMetrics)

is *numerically identical* to the official Kaggle / cellcanvas reference in

    3DINO/dinov2/eval/detection_3d/official_metrics_czii.py   (score())

The official `score()` is DataFrame-based and has no notion of confidence
thresholds. Our `CZIDetectionMetrics` stores per-tomogram predictions and
sweeps a score threshold. To compare apples-to-apples we:

  * give every prediction a confidence of 1.0,
  * evaluate our metric at threshold 0.0 (i.e. keep every prediction),
  * feed coordinates already in Angstrom with voxel_size = 1.0,

so that the only thing left to differ is the actual F-beta math.

Three independent checks are run:

  1. CONSTANTS  - particle radii & per-class weights agree (by order/name).
  2. compute_metrics - the verbatim KDTree (tp, fp, fn) helper in both files
                       returns identical results on random point clouds.
  3. AGGREGATE  - the final weighted F-beta agrees across many random
                  multi-tomogram scenes and several (distance_multiplier, beta)
                  settings.

Exit code is 0 iff every check passes.

Run:
    python scripts_debug/compare_czii_metrics.py
    python scripts_debug/compare_czii_metrics.py --trials 500 --verbose
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load the two modules directly by file path (avoids importing the whole
# dinov2 package, which has heavy deps). Both files only need numpy/scipy/pandas.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DET_DIR = os.path.join(HERE, "..", "3DINO", "dinov2", "eval", "detection_3d")
DET_DIR = os.path.normpath(DET_DIR)

OURS_PATH = os.path.join(DET_DIR, "metrics.py")
OFFICIAL_PATH = os.path.join(DET_DIR, "official_metrics_czii.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ours = _load(OURS_PATH, "cryodino_metrics_ours")
official = _load(OFFICIAL_PATH, "cryodino_metrics_official")


# Class index -> official particle_type name. Order matches
# CZIDetectionMetrics.PARTICLE_NAMES / PARTICLE_RADII_ANG / WEIGHTS, which are now
# ALPHABETICAL by our particle spelling (= the model's class-id convention, set by
# load_detection_annotations). Official spellings map as:
#   Beta-amylase=beta-amylase, Beta-galactosidase=beta-galactosidase,
#   Thyroglobulin=thyroglobulin, cytosolic ribosome=ribosome,
#   ferritin complex=apo-ferritin, virus-like capsid=virus-like-particle.
OFFICIAL_NAMES = [
    "beta-amylase",          # 0  Beta-amylase
    "beta-galactosidase",    # 1  Beta-galactosidase
    "thyroglobulin",         # 2  Thyroglobulin
    "ribosome",              # 3  cytosolic ribosome
    "apo-ferritin",          # 4  ferritin complex
    "virus-like-particle",   # 5  virus-like capsid
]

# The official score() reads radii & weights from these dicts.
OFFICIAL_RADII = {
    "apo-ferritin": 60,
    "beta-amylase": 65,
    "beta-galactosidase": 90,
    "ribosome": 150,
    "thyroglobulin": 130,
    "virus-like-particle": 135,
}
OFFICIAL_WEIGHTS = {
    "apo-ferritin": 1,
    "beta-amylase": 0,
    "beta-galactosidase": 2,
    "ribosome": 1,
    "thyroglobulin": 2,
    "virus-like-particle": 1,
}
NUM_CLASSES = len(OFFICIAL_NAMES)


# ---------------------------------------------------------------------------
# Check 1: constants
# ---------------------------------------------------------------------------
def check_constants():
    print("=" * 70)
    print("CHECK 1 - constants (radii & weights)")
    print("=" * 70)
    ok = True

    radii_ours = list(ours.CZIDetectionMetrics.PARTICLE_RADII_ANG)
    radii_off = [OFFICIAL_RADII[n] for n in OFFICIAL_NAMES]
    print(f"  radii   ours    : {radii_ours}")
    print(f"  radii   official: {radii_off}")
    if radii_ours != radii_off:
        print("  [FAIL] particle radii differ")
        ok = False

    w_ours = list(ours.CZIDetectionMetrics.WEIGHTS)
    w_off = [OFFICIAL_WEIGHTS[n] for n in OFFICIAL_NAMES]
    print(f"  weights ours    : {w_ours}")
    print(f"  weights official: {w_off}")
    if w_ours != w_off:
        print("  [FAIL] weights differ")
        ok = False

    print(f"  [{'PASS' if ok else 'FAIL'}] constants\n")
    return ok


# ---------------------------------------------------------------------------
# Check 2: compute_metrics helper (defined separately in each file)
# ---------------------------------------------------------------------------
def check_compute_metrics(trials, rng, verbose):
    print("=" * 70)
    print("CHECK 2 - compute_metrics(ref, radius, cand) tp/fp/fn equivalence")
    print("=" * 70)
    mismatches = 0
    for t in range(trials):
        n_ref = rng.integers(0, 30)
        n_cand = rng.integers(0, 30)
        ref = rng.uniform(0, 500, size=(n_ref, 3))
        cand = rng.uniform(0, 500, size=(n_cand, 3))
        radius = float(rng.uniform(1, 80))

        a = ours.compute_metrics(ref, radius, cand)
        b = official.compute_metrics(ref, radius, cand)
        if a != b:
            mismatches += 1
            if verbose:
                print(f"  trial {t}: ours={a} official={b}")
    ok = mismatches == 0
    print(f"  ran {trials} random trials, {mismatches} mismatch(es)")
    print(f"  [{'PASS' if ok else 'FAIL'}] compute_metrics\n")
    return ok


# ---------------------------------------------------------------------------
# Check 3: full aggregate F-beta
# ---------------------------------------------------------------------------
def _make_scene(rng, n_tomos):
    """Build a random multi-tomogram scene.

    Returns:
        tomos: list of dicts with gt (Nx4: x,y,z,cls) and pred (Mx4: x,y,z,cls)
    Coordinates are continuous floats so no duplicate (x,y,z) rows occur,
    which the official score() asserts against.
    """
    tomos = []
    for _ in range(n_tomos):
        gt_xyz, gt_cls = [], []
        pr_xyz, pr_cls = [], []
        for c in range(NUM_CLASSES):
            n_gt = rng.integers(0, 8)
            for _ in range(n_gt):
                gt_xyz.append(rng.uniform(0, 1000, size=3))
                gt_cls.append(c)
            n_pr = rng.integers(0, 8)
            for _ in range(n_pr):
                pr_xyz.append(rng.uniform(0, 1000, size=3))
                pr_cls.append(c)
        tomos.append({
            "gt_xyz": np.array(gt_xyz).reshape(-1, 3),
            "gt_cls": np.array(gt_cls, dtype=int).reshape(-1),
            "pr_xyz": np.array(pr_xyz).reshape(-1, 3),
            "pr_cls": np.array(pr_cls, dtype=int).reshape(-1),
        })
    return tomos


def _official_score(tomos, distance_multiplier, beta):
    sol_rows, sub_rows = [], []
    for i, tm in enumerate(tomos):
        exp = f"exp{i}"
        for (x, y, z), c in zip(tm["gt_xyz"], tm["gt_cls"]):
            sol_rows.append((exp, OFFICIAL_NAMES[c], x, y, z))
        for (x, y, z), c in zip(tm["pr_xyz"], tm["pr_cls"]):
            sub_rows.append((exp, OFFICIAL_NAMES[c], x, y, z))

    cols = ["experiment", "particle_type", "x", "y", "z"]
    solution = pd.DataFrame(sol_rows, columns=cols)
    submission = pd.DataFrame(sub_rows, columns=cols)
    # add an id column (score() takes the name but does not use values)
    solution["id"] = np.arange(len(solution))
    submission["id"] = np.arange(len(submission))

    return official.score(
        solution=solution,
        submission=submission,
        row_id_column_name="id",
        distance_multiplier=distance_multiplier,
        beta=beta,
    )


def _ours_score(tomos, distance_multiplier, beta):
    metric = ours.CZIDetectionMetrics(distance_multiplier=distance_multiplier, beta=beta)
    for tm in tomos:
        n = len(tm["pr_xyz"])
        pred_centers = tm["pr_xyz"]                      # already Angstrom
        pred_labels = tm["pr_cls"]
        pred_scores = np.ones(n, dtype=np.float32)        # keep all at thr=0
        # gt_points: [N, 5] = x, y, z, class_id, sigma(unused)
        m = len(tm["gt_xyz"])
        gt_points = np.zeros((m, 5), dtype=np.float64)
        if m:
            gt_points[:, :3] = tm["gt_xyz"]
            gt_points[:, 3] = tm["gt_cls"]
        metric.accumulate(
            pred_centers_vox=pred_centers,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            gt_points=gt_points,
            voxel_size=1.0,                               # Angstrom == voxel
        )
    agg_f4, _ = metric._f4_at_threshold(0.0)              # keep everything
    return agg_f4


def check_aggregate(trials, rng, verbose, atol):
    print("=" * 70)
    print("CHECK 3 - aggregate weighted F-beta equivalence")
    print("=" * 70)
    settings = [
        (0.5, 4),   # CZIDetectionMetrics defaults
        (0.5, 1),
        (1.0, 4),
        (0.8, 2),
    ]
    mismatches = 0
    worst = 0.0
    n_run = 0
    for t in range(trials):
        n_tomos = int(rng.integers(1, 5))
        tomos = _make_scene(rng, n_tomos)
        for dm, beta in settings:
            s_off = _official_score(tomos, dm, beta)
            s_ours = _ours_score(tomos, dm, beta)
            diff = abs(s_off - s_ours)
            worst = max(worst, diff)
            n_run += 1
            if diff > atol:
                mismatches += 1
                if verbose:
                    print(f"  trial {t} dm={dm} beta={beta}: "
                          f"official={s_off:.12f} ours={s_ours:.12f} diff={diff:.2e}")
    ok = mismatches == 0
    print(f"  ran {n_run} comparisons over {trials} random scenes x "
          f"{len(settings)} settings")
    print(f"  max abs diff = {worst:.3e}  (tolerance {atol:.0e})")
    print(f"  {mismatches} mismatch(es)")
    print(f"  [{'PASS' if ok else 'FAIL'}] aggregate F-beta\n")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=200,
                    help="random trials per check (default 200)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atol", type=float, default=1e-12,
                    help="absolute tolerance for the aggregate F-beta (default 1e-12)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"ours     : {OURS_PATH}")
    print(f"official : {OFFICIAL_PATH}\n")

    rng = np.random.default_rng(args.seed)
    results = [
        check_constants(),
        check_compute_metrics(args.trials, rng, args.verbose),
        check_aggregate(args.trials, rng, args.verbose, args.atol),
    ]

    print("=" * 70)
    all_ok = all(results)
    if all_ok:
        print("RESULT: PASS - the two CZII metrics are numerically identical.")
    else:
        print("RESULT: FAIL - the two CZII metrics DIFFER (see above).")
    print("=" * 70)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
