"""
Scan a MONAI cache_dir for corrupted .pt files and delete them.
PyTorch saves are zip archives; opening one as ZipFile checks the central
directory instantly without reading tensor data.

Usage:
    python utils/fix_cache.py --cache-dir <path> [--dry-run] [--workers 32]
"""

import argparse
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path


def _check(path: str):
    try:
        with zipfile.ZipFile(path, "r"):
            pass
        return None
    except Exception:
        return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", help="report but don't delete")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    print(f"Scanning {cache_dir} ...")

    files = [str(p) for p in cache_dir.rglob("*.pt")]
    total = len(files)
    print(f"Found {total:,} .pt files — checking with {args.workers} workers")

    corrupted = []
    batch = 10_000
    for i in range(0, total, batch):
        chunk = files[i : i + batch]
        with Pool(args.workers) as pool:
            results = pool.map(_check, chunk)
        bad = [r for r in results if r is not None]
        corrupted.extend(bad)
        done = min(i + batch, total)
        print(f"  {done:,}/{total:,} checked — {len(corrupted)} corrupted so far")

    print(f"\nTotal corrupted: {len(corrupted)}")
    for f in corrupted:
        print(f"  {f}")

    if not corrupted:
        print("Cache is clean.")
        return

    if args.dry_run:
        print("\nDry-run mode — nothing deleted.")
    else:
        for f in corrupted:
            os.remove(f)
        print(f"\nDeleted {len(corrupted)} corrupted file(s).")


if __name__ == "__main__":
    main()
