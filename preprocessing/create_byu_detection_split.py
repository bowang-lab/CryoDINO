"""Create a motor-count-stratified train/val/test split + point annotations CSV for BYU.

Reads the Kaggle BYU train_labels.csv and writes two files consumed by
downstream_patch_generation.py --detection:

  1. point_annotations.csv  (run, particle_name, z, y, x, voxel_size)
     - Motor axis 0/1/2 → z/y/x (voxels). jpg_to_nifti.py stacks slices as (z, y, x)
       via SimpleITK, so nibabel loads the volume as (X, Y, Z) and the patcher's
       CSV (z, y, x) → (x, y, z) reorder lines up.
     - No-motor rows (coords = -1) are dropped.

  2. byu_split_datalist.json  {"training", "validation", "test"}
     - Split at tomogram level, stratified by motor count bin (0 / 1 / 2+):
       each bin is shuffled with --seed and divided by the requested fractions.
     - Entries: {"image": <nifti path>, "voxel_size": <Å/vox>, "num_motors": <int>}.
       voxel_size is per tomogram so no-motor tomograms (absent from the CSV)
       still get their true spacing.

Usage:
    python preprocessing/create_byu_detection_split.py \\
        --labels-csv /media/sumin/T9/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv \\
        --images-dir /media/sumin/T9/byu-locating-bacterial-flagellar-motors-2025/nifti/train \\
        --output-dir /media/sumin/T9/byu_detection \\
        --train 70 --val 15 --test 15 --seed 42
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PARTICLE_NAME = "motor"


def count_bin(num_motors: int) -> str:
    return str(num_motors) if num_motors < 2 else "2+"


def load_labels(labels_csv: Path) -> dict:
    """Return {tomo_id: {'voxel_size': float, 'num_motors': int, 'points': [(z, y, x), ...]}}."""
    tomos = {}
    with open(labels_csv, "r") as f:
        for row in csv.DictReader(f):
            tomo_id = row["tomo_id"].strip()
            info = tomos.setdefault(tomo_id, {
                "voxel_size": float(row["Voxel spacing"]),
                "num_motors": int(row["Number of motors"]),
                "points":     [],
            })
            z, y, x = (float(row[f"Motor axis {i}"]) for i in range(3))
            if min(z, y, x) >= 0:
                info["points"].append((z, y, x))
    return tomos


def stratified_split(tomo_ids: list, tomos: dict, fractions: tuple, seed: int) -> tuple:
    """Shuffle each motor-count bin and split it by fractions; remainder goes to test."""
    by_bin = defaultdict(list)
    for tomo_id in sorted(tomo_ids):
        by_bin[count_bin(tomos[tomo_id]["num_motors"])].append(tomo_id)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for bin_name in sorted(by_bin):
        ids = by_bin[bin_name]
        rng.shuffle(ids)
        n_train = round(len(ids) * fractions[0])
        n_val   = round(len(ids) * fractions[1])
        train += ids[:n_train]
        val   += ids[n_train:n_train + n_val]
        test  += ids[n_train + n_val:]
    return train, val, test


def write_annotations(out_path: Path, tomo_ids: list, tomos: dict):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "particle_name", "z", "y", "x", "voxel_size"])
        for tomo_id in sorted(tomo_ids):
            info = tomos[tomo_id]
            for z, y, x in info["points"]:
                writer.writerow([tomo_id, PARTICLE_NAME, z, y, x, info["voxel_size"]])


def print_distribution(splits: dict, tomos: dict):
    bins = ["0", "1", "2+"]
    print(f"{'split':<12}{'n':>5}" + "".join(f"{b:>8}" for b in bins) + f"{'motors':>9}")
    for name, ids in splits.items():
        counts = Counter(count_bin(tomos[t]["num_motors"]) for t in ids)
        n_motors = sum(len(tomos[t]["points"]) for t in ids)
        pct = "".join(f"{100 * counts[b] / max(len(ids), 1):>7.1f}%" for b in bins)
        print(f"{name:<12}{len(ids):>5}{pct}{n_motors:>9}")


def main():
    parser = argparse.ArgumentParser(description="BYU motor-count-stratified split + point annotations CSV.")
    parser.add_argument("--labels-csv", type=Path, required=True, help="Kaggle train_labels.csv")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory of <tomo_id>.nii.gz")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write the CSV and split JSON")
    parser.add_argument("--train", type=float, default=70, help="Train %% (default: 70)")
    parser.add_argument("--val",   type=float, default=15, help="Val %% (default: 15)")
    parser.add_argument("--test",  type=float, default=15, help="Test %% (default: 15)")
    parser.add_argument("--seed",  type=int,   default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 100) > 0.01:
        raise ValueError(f"Percentages must sum to 100, got {total}")

    tomos = load_labels(args.labels_csv)

    image_paths = {}
    for tomo_id in tomos:
        img_path = args.images_dir / f"{tomo_id}.nii.gz"
        if img_path.exists():
            image_paths[tomo_id] = img_path.resolve()
        else:
            print(f"WARNING: no NIfTI for {tomo_id}, skipping")
    if not image_paths:
        raise ValueError(f"No NIfTI files matched train_labels.csv in {args.images_dir}")

    train, val, test = stratified_split(
        list(image_paths), tomos, (args.train / 100, args.val / 100), args.seed
    )
    splits = {"training": train, "validation": val, "test": test}

    def to_entry(tomo_id):
        return {
            "image":      str(image_paths[tomo_id]),
            "voxel_size": tomos[tomo_id]["voxel_size"],
            "num_motors": tomos[tomo_id]["num_motors"],
        }

    datalist = {name: [to_entry(t) for t in sorted(ids)] for name, ids in splits.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = args.output_dir / "point_annotations.csv"
    json_path = args.output_dir / "byu_split_datalist.json"
    write_annotations(csv_path, list(image_paths), tomos)
    json_path.write_text(json.dumps(datalist, indent=2))

    print_distribution(splits, tomos)
    print(f"\nAnnotations: {csv_path}")
    print(f"Datalist   : {json_path}")


if __name__ == "__main__":
    main()
