"""
Script to validate .pt patch files for NaN values and dtype issues.
Usage: python validate_patches.py --json pretrain.json --workers 16
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def validate_patch(path: str) -> tuple[str, dict]:
    """Check a single patch for issues. Returns (path, issues_dict)."""
    issues = {}

    if not Path(path).exists():
        return path, {"missing": True}

    try:
        # tensor = torch.load(path, map_location='cpu', weights_only=True)
        tensor = torch.load(path)
    except Exception as e:
        return path, {"load_error": str(e)}

    # Check for NaN
    nan_count = torch.isnan(tensor).sum().item()
    if nan_count > 0:
        issues["nan_count"] = nan_count
        issues["nan_percent"] = 100 * nan_count / tensor.numel()

    # Check for Inf
    inf_count = torch.isinf(tensor).sum().item()
    if inf_count > 0:
        issues["inf_count"] = inf_count

    # Check dtype
    if tensor.dtype not in [torch.float32, torch.float64]:
        issues["bad_dtype"] = str(tensor.dtype)

    # Check for extreme values
    if tensor.numel() > 0 and not torch.isnan(tensor).all():
        valid_tensor = tensor[~torch.isnan(tensor)]
        if valid_tensor.numel() > 0:
            min_val = valid_tensor.min().item()
            max_val = valid_tensor.max().item()
            if abs(min_val) > 1e6 or abs(max_val) > 1e6:
                issues["extreme_values"] = {"min": min_val, "max": max_val}

    issues["dtype"] = str(tensor.dtype)
    issues["shape"] = list(tensor.shape)

    return path, issues


def main():
    parser = argparse.ArgumentParser(description="Validate .pt patches for NaN/Inf/dtype issues")
    parser.add_argument("--json", type=str, required=True, help="Path to pretrain.json")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to check")
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        data = json.load(f)

    paths = [item["image"] for item in data]
    if args.limit:
        paths = paths[:args.limit]

    print(f"Checking {len(paths)} patches with {args.workers} workers...\n")

    # Track stats
    problematic = []
    dtype_counts = {}
    missing = 0
    checked = 0

    # Parallel processing with immediate output
    with Pool(args.workers) as pool:
        for path, issues in tqdm(pool.imap_unordered(validate_patch, paths), total=len(paths)):
            checked += 1

            if issues.get("missing"):
                missing += 1
                tqdm.write(f"[MISSING] {path}")
                continue

            dtype = issues.get("dtype", "unknown")
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

            has_issues = any(k in issues for k in ["nan_count", "inf_count", "bad_dtype", "extreme_values", "load_error"])
            if has_issues:
                problematic.append({"path": path, **issues})
                # Print immediately
                msg = f"[PROBLEM] {path}"
                if "nan_count" in issues:
                    msg += f" | NaN: {issues['nan_count']} ({issues['nan_percent']:.2f}%)"
                if "inf_count" in issues:
                    msg += f" | Inf: {issues['inf_count']}"
                if "load_error" in issues:
                    msg += f" | Error: {issues['load_error']}"
                tqdm.write(msg)

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Total: {len(paths)} | Missing: {missing} | Problematic: {len(problematic)}")
    print(f"Dtype distribution: {dtype_counts}")

    if problematic:
        with open("problematic_patches.json", 'w') as f:
            json.dump(problematic, f, indent=2)
        print(f"Full list saved to: problematic_patches.json")


if __name__ == "__main__":
    main()
