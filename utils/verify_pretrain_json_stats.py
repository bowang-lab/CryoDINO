"""
Randomly sample patches from pretrain.json and verify that stored tomo_mean/tomo_std
match what's computed directly from the corresponding NIfTI file.

Usage:
    python utils/verify_pretrain_json_stats.py \
        --json /path/to/pretrain.json \
        --n 50 \
        [--tol 1e-4]
"""

import argparse
import json
import random
import re
import sys

import nibabel as nib
import numpy as np


def pt_path_to_nifti(pt_path: str) -> str:
    nifti_path = pt_path.replace("_subtomograms", "")
    nifti_path = re.sub(r"_patch_.*\.pt$", ".nii.gz", nifti_path)
    return nifti_path


def compute_stats(nifti_path: str):
    img = nib.load(nifti_path)
    data = img.get_fdata(dtype=np.float32)
    return float(data.mean()), float(data.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to pretrain.json")
    parser.add_argument("--n", type=int, default=50, help="Number of patches to sample")
    parser.add_argument("--tol", type=float, default=1e-4, help="Absolute tolerance for comparison")
    args = parser.parse_args()

    with open(args.json) as f:
        dataset = json.load(f)

    # Filter entries that have the stats fields
    has_stats = [e for e in dataset if "tomo_mean" in e and "tomo_std" in e]
    missing = len(dataset) - len(has_stats)
    if missing:
        print(f"Warning: {missing}/{len(dataset)} entries missing tomo_mean/tomo_std — sampling from the {len(has_stats)} that have them")

    if not has_stats:
        print("No entries have tomo_mean/tomo_std. Run create_pretrain_json.py --update-existing first.")
        sys.exit(1)

    n = min(args.n, len(has_stats))
    samples = random.sample(has_stats, n)

    # Deduplicate by nifti so we don't reload the same tomogram many times
    nifti_map: dict[str, list] = {}
    for entry in samples:
        nifti_path = pt_path_to_nifti(entry["image"])
        nifti_map.setdefault(nifti_path, []).append(entry)

    print(f"Sampled {n} patches from {len(nifti_map)} unique tomograms\n")

    failures = []
    for i, (nifti_path, entries) in enumerate(nifti_map.items(), 1):
        print(f"[{i}/{len(nifti_map)}] {nifti_path}")
        try:
            true_mean, true_std = compute_stats(nifti_path)
        except Exception as e:
            print(f"  ERROR loading NIfTI: {e}")
            for entry in entries:
                failures.append((entry["image"], "nifti_load_error"))
            continue

        for entry in entries:
            stored_mean = entry["tomo_mean"]
            stored_std = entry["tomo_std"]
            mean_diff = abs(stored_mean - true_mean)
            std_diff = abs(stored_std - true_std)
            ok = mean_diff <= args.tol and std_diff <= args.tol
            status = "OK" if ok else "MISMATCH"
            print(f"  {status}  patch={entry['image'].split('/')[-1]}"
                  f"  mean: stored={stored_mean:.6f} true={true_mean:.6f} diff={mean_diff:.2e}"
                  f"  std: stored={stored_std:.6f} true={true_std:.6f} diff={std_diff:.2e}")
            if not ok:
                failures.append((entry["image"], f"mean_diff={mean_diff:.2e} std_diff={std_diff:.2e}"))

    print(f"\n{'='*60}")
    if failures:
        print(f"FAILED: {len(failures)} / {n} patches")
        for patch, reason in failures:
            print(f"  {patch}: {reason}")
        sys.exit(1)
    else:
        print(f"All {n} patches passed (tol={args.tol})")


if __name__ == "__main__":
    main()
