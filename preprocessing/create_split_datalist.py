"""Create train/val/test split datalist JSON for an nnUNet-style dataset.
  cd projects/CryoDINO
  python preprocessing/create_split_datalist.py \
      /cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream_detection/czi_dataset/Dataset440_CZII_10440 \
      --detection \
      --val-dir  /cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream_detection/czi_dataset/Dataset445_CZII_10445 \
      --test-dir /cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream_detection/czi_dataset/Dataset446_CZII_10446 \
      --output   /cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream_detection/czi_dataset/Dataset440_CZII_10440/czi_split_datalist.json


Adapted from CryoET/dataset/downstream/create_datalist.py.
Adds --detection flag for datasets without labelsTr/ (image-only entries).
Adds --val-dir / --test-dir to source splits from separate dataset directories.

Usage — detection, splits across three datasets:
    python preprocessing/create_split_datalist.py Dataset440_CZII_10440 \\
        --detection \\
        --val-dir  Dataset445_CZII_10445 \\
        --test-dir Dataset446_CZII_10446 \\
        --output   /path/to/czi_split_datalist.json

Usage — detection, single dataset with fraction-based split:
    python preprocessing/create_split_datalist.py Dataset440_CZII_10440 \\
        --detection --train 80 --val 20 --test 0 --seed 42

Usage — segmentation (image + label), single dataset:
    python preprocessing/create_split_datalist.py /path/to/DatasetXXX_Name \\
        --train 70 --val 15 --test 15 --seed 42
"""

import argparse
import json
import random
import re
from pathlib import Path


def collect_entries(images_dir: Path, labels_dir: Path, detection: bool) -> list:
    entries = []
    for img_path in sorted(images_dir.glob("*.nii.gz")):
        if detection:
            entries.append({"image": str(img_path.resolve())})
        else:
            case_id    = re.sub(r"_\d{4}(\.nii\.gz)$", r"\1", img_path.name)
            label_path = labels_dir / case_id
            if label_path.exists():
                entries.append({
                    "image": str(img_path.resolve()),
                    "label": str(label_path.resolve()),
                })
            else:
                print(f"WARNING: no matching label for {img_path.name}, skipping")
    return entries


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test datalist JSON.")
    parser.add_argument("dataset_dir", type=Path,
                        help="Training dataset directory (contains imagesTr/).")
    parser.add_argument("--val-dir",  type=Path, default=None,
                        help="Separate dataset directory for validation split.")
    parser.add_argument("--test-dir", type=Path, default=None,
                        help="Separate dataset directory for test split.")
    parser.add_argument("--train", type=float, default=70,
                        help="Train %% for fraction-based split (default: 70). Ignored when --val-dir is given.")
    parser.add_argument("--val",   type=float, default=15,
                        help="Val %% for fraction-based split (default: 15). Ignored when --val-dir is given.")
    parser.add_argument("--test",  type=float, default=15,
                        help="Test %% for fraction-based split (default: 15). Ignored when --test-dir is given.")
    parser.add_argument("--seed",  type=int, default=None,
                        help="Random seed for fraction-based split (default: None).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: <dataset_dir>/<name>_100_datalist.json).")
    parser.add_argument("--detection", action="store_true",
                        help="Detection mode: entries have only 'image' key (no labelsTr needed).")
    args = parser.parse_args()

    def get_dirs(dataset_dir):
        return dataset_dir / "imagesTr", dataset_dir / "labelsTr"

    # --- training entries ---------------------------------------------------
    images_dir, labels_dir = get_dirs(args.dataset_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"imagesTr not found in {args.dataset_dir}")
    if not args.detection and not labels_dir.is_dir():
        raise FileNotFoundError(f"labelsTr not found in {args.dataset_dir}. Use --detection for image-only datasets.")

    all_train = collect_entries(images_dir, labels_dir, args.detection)
    if not all_train:
        raise ValueError(f"No valid entries found in {images_dir}")

    # --- separate-dir splits ------------------------------------------------
    if args.val_dir or args.test_dir:
        train_data = all_train
        val_data   = []
        test_data  = []

        if args.val_dir:
            vi, vl = get_dirs(args.val_dir)
            if not vi.is_dir():
                raise FileNotFoundError(f"imagesTr not found in {args.val_dir}")
            val_data = collect_entries(vi, vl, args.detection)

        if args.test_dir:
            ti, tl = get_dirs(args.test_dir)
            if not ti.is_dir():
                raise FileNotFoundError(f"imagesTr not found in {args.test_dir}")
            test_data = collect_entries(ti, tl, args.detection)

    # --- fraction-based split (single dataset) ------------------------------
    else:
        total = args.train + args.val + args.test
        if abs(total - 100) > 0.01:
            raise ValueError(f"Percentages must sum to 100, got {total}")

        if args.seed is not None:
            random.seed(args.seed)
        random.shuffle(all_train)

        n       = len(all_train)
        n_train = int(n * args.train / 100)
        n_val   = int(n * args.val   / 100)

        train_data = all_train[:n_train]
        val_data   = all_train[n_train:n_train + n_val]
        test_data  = all_train[n_train + n_val:]

    datalist = {
        "training":   train_data,
        "validation": val_data,
        "test":       test_data,
    }

    dataset_name = args.dataset_dir.name
    out_path = args.output or (args.dataset_dir / f"{dataset_name}_100_datalist.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(datalist, indent=2))

    print(f"Created datalist: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
