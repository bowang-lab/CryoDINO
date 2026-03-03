"""
Check patches for constant/near-constant intensity (percentile issues).
Usage: python check_percentiles.py --json pretrain.json --workers 16
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple, Dict


def check_patch(path: str) -> Tuple[str, Optional[Dict]]:
    """Check if 2nd and 98th percentiles are too close."""
    if not Path(path).exists():
        return path, {"error": "missing"}

    try:
        tensor = torch.load(path).float().flatten()
    except Exception as e:
        return path, {"error": str(e)}

    # Check p2 vs p98 diff
    p2 = torch.quantile(tensor, 0.02).item()
    p98 = torch.quantile(tensor, 0.98).item()
    diff = abs(p98 - p2)

    if diff < 1e-4:  # Too close - will cause NaN
        # Compute all other percentiles for reporting
        p1 = torch.quantile(tensor, 0.01).item()
        p99 = torch.quantile(tensor, 0.99).item()
        p05 = torch.quantile(tensor, 0.005).item()
        p995 = torch.quantile(tensor, 0.995).item()
        p0005 = torch.quantile(tensor, 0.0005).item()
        p9995 = torch.quantile(tensor, 0.9995).item()
        p0002 = torch.quantile(tensor, 0.0002).item()
        p9998 = torch.quantile(tensor, 0.9998).item()

        return path, {
            "p2": p2, "p98": p98, "diff": diff,
            "p1": p1, "p99": p99,
            "p05": p05, "p995": p995,
            "p0005": p0005, "p9995": p9995,
            "p0002": p0002, "p9998": p9998,
            "std": tensor.std().item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
        }

    return path, None  # OK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--workers", type=int, default=cpu_count())
    parser.add_argument("--threshold", type=float, default=1e-4, help="Min diff between p2 and p98")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        data = json.load(f)

    paths = [item["image"] for item in data]
    if args.limit:
        paths = paths[:args.limit]

    print(f"Checking {len(paths)} patches with {args.workers} workers...\n")

    problematic = []

    with Pool(args.workers) as pool:
        for path, result in tqdm(pool.imap_unordered(check_patch, paths), total=len(paths)):
            if result is not None:
                problematic.append({"path": path, **result})
                if "error" in result:
                    tqdm.write(f"[ERROR] {path} | {result['error']}")
                else:
                    tqdm.write(f"[BAD] {path}")
                    tqdm.write(f"  p2={result['p2']:.6f} p98={result['p98']:.6f} | p1={result['p1']:.6f} p99={result['p99']:.6f}")
                    tqdm.write(f"  p05={result['p05']:.6f} p995={result['p995']:.6f} | p0005={result['p0005']:.6f} p9995={result['p9995']:.6f}")
                    tqdm.write(f"  p0002={result['p0002']:.6f} p9998={result['p9998']:.6f}")
                    tqdm.write(f"  min={result['min']:.6f} max={result['max']:.6f} std={result['std']:.2e} diff={result['diff']:.2e}")

    print(f"\n{'='*60}")
    print(f"Found {len(problematic)} problematic patches out of {len(paths)}")

    if problematic:
        with open("bad_percentile_patches.json", 'w') as f:
            json.dump(problematic, f, indent=2)
        print(f"Saved to: bad_percentile_patches.json")

        # Print paths only for easy filtering
        with open("bad_percentile_paths.txt", 'w') as f:
            for item in problematic:
                f.write(item["path"] + "\n")
        print(f"Paths saved to: bad_percentile_paths.txt")


if __name__ == "__main__":
    main()
