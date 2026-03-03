"""
Check whole tomograms for constant/near-constant intensity (percentile issues).
Usage: python check_tomogram_percentiles.py --folders /path/to/folder1 /path/to/folder2 --workers 16
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple, Dict, List
import nibabel as nib


def load_tomogram(path: Path) -> Optional[np.ndarray]:
    """Load tomogram from various formats."""
    suffix = path.suffix.lower()

    if suffix in ['.gz', '.nii']:
        # Handle .nii.gz
        img = nib.load(str(path))
        return img.get_fdata()
    elif suffix in ['.tif', '.tiff']:
        from tifffile import imread
        return imread(str(path))
    elif suffix == '.mrc':
        import mrcfile
        with mrcfile.open(str(path), permissive=True) as mrc:
            return mrc.data.copy()
    elif suffix == '.pt':
        return torch.load(str(path)).numpy()
    else:
        return None


def check_tomogram(path: str) -> Tuple[str, Optional[Dict]]:
    """Check if 2nd and 98th percentiles are too close."""
    path = Path(path)

    if not path.exists():
        return str(path), {"error": "missing"}

    try:
        data = load_tomogram(path)
        if data is None:
            return str(path), {"error": f"unsupported format: {path.suffix}"}

        arr = data.astype(np.float32).flatten()
    except Exception as e:
        return str(path), {"error": str(e)}

    # Compute all percentiles
    p2 = np.percentile(arr, 2).item()
    p98 = np.percentile(arr, 98).item()
    diff = abs(p98 - p2)
    p1 = np.percentile(arr, 1).item()
    p99 = np.percentile(arr, 99).item()
    p05 = np.percentile(arr, 0.5).item()
    p995 = np.percentile(arr, 99.5).item()
    p0005 = np.percentile(arr, 0.05).item()
    p9995 = np.percentile(arr, 99.95).item()
    p0002 = np.percentile(arr, 0.02).item()
    p9998 = np.percentile(arr, 99.98).item()

    return str(path), {
        "p2": p2, "p98": p98, "diff": diff,
        "p1": p1, "p99": p99,
        "p05": p05, "p995": p995,
        "p0005": p0005, "p9995": p9995,
        "p0002": p0002, "p9998": p9998,
        "std": arr.std().item(),
        "min": arr.min().item(),
        "max": arr.max().item(),
        "shape": list(data.shape),
    }


def find_tomograms(folders: List[str], extensions: List[str]) -> List[str]:
    """Find all tomogram files in folders."""
    files = []
    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            print(f"Warning: folder not found: {folder}")
            continue
        for ext in extensions:
            files.extend([str(f) for f in folder.rglob(f"*{ext}")])
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", type=str, nargs="+", required=True, help="Folders containing tomograms")
    parser.add_argument("--workers", type=int, default=cpu_count())
    parser.add_argument("--extensions", type=str, nargs="+", default=[".nii.gz", ".nii", ".tif", ".tiff", ".mrc"],
                        help="File extensions to search for")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Find all tomogram files
    paths = find_tomograms(args.folders, args.extensions)

    if not paths:
        print("No tomogram files found!")
        return

    if args.limit:
        paths = paths[:args.limit]

    print(f"Found {len(paths)} tomograms")
    print(f"Checking with {args.workers} workers...\n")

    results = []

    with Pool(args.workers) as pool:
        for path, result in tqdm(pool.imap_unordered(check_tomogram, paths), total=len(paths)):
            if result is not None:
                results.append({"path": path, **result})
                if "error" in result:
                    tqdm.write(f"[ERROR] {path} | {result['error']}")
                else:
                    tqdm.write(f"[OK] {path}")
                    tqdm.write(f"  shape={result['shape']}")
                    tqdm.write(f"  p2={result['p2']:.6f} p98={result['p98']:.6f} | p1={result['p1']:.6f} p99={result['p99']:.6f}")
                    tqdm.write(f"  p05={result['p05']:.6f} p995={result['p995']:.6f} | p0005={result['p0005']:.6f} p9995={result['p9995']:.6f}")
                    tqdm.write(f"  p0002={result['p0002']:.6f} p9998={result['p9998']:.6f}")
                    tqdm.write(f"  min={result['min']:.6f} max={result['max']:.6f} std={result['std']:.2e} diff={result['diff']:.2e}")

    print(f"\n{'='*60}")
    print(f"Processed {len(results)} tomograms out of {len(paths)}")

    if results:
        with open("tomogram_percentiles.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to: tomogram_percentiles.json")


if __name__ == "__main__":
    main()
