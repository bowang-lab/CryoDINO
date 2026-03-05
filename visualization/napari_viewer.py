"""
Standalone napari viewer for saved segmentation predictions.

Opens an interactive 3D viewer to browse (image, prediction, ground-truth)
triplets without requiring a GPU or the segmentation model.

Usage:
    python visualization/napari_viewer.py \
        --pred-dir /path/to/output/ \
        --image-dir /path/to/images/ \
        --label-dir /path/to/labels/     # optional
"""

import argparse
import os
import sys

# Ensure project root is on the path so we can import visualization utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from visualization._napari_utils import load_samples_from_output_dir, launch_viewer


def main():
    parser = argparse.ArgumentParser(
        description="View saved segmentation predictions in napari (no GPU needed)."
    )
    parser.add_argument(
        "--pred-dir", type=str, required=True,
        help="Directory containing predicted segmentation NIfTI files",
    )
    parser.add_argument(
        "--image-dir", type=str, required=True,
        help="Directory containing the raw input NIfTI images",
    )
    parser.add_argument(
        "--label-dir", type=str, default=None,
        help="Optional directory containing ground-truth label NIfTI files",
    )
    parser.add_argument(
        "--metrics", type=str, default=None,
        help="Path to metrics.json (default: <pred-dir>/metrics.json)",
    )
    args = parser.parse_args()

    samples = load_samples_from_output_dir(
        pred_dir=args.pred_dir,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        metrics_path=args.metrics,
    )
    launch_viewer(samples)


if __name__ == "__main__":
    main()
