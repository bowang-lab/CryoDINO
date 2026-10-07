#!/bin/bash
# =========================
# Detection patch generation (BYU flagellar motors) — local run, outputs on T9
#   Step 1: train_labels.csv → point_annotations.csv + motor-count-stratified
#           70/15/15 split (byu_split_datalist.json)
#   Step 2: downstream_patch_generation.py --detection
#           → overlapping 128^3 .pt patches (all kept) + byu_100_datalist.json
# Native voxel spacing (6.5–19.7 Å/vox); sigma converted per tomogram (200 Å / spacing).
# Disk: ~480k train patches × 8 MB (float32 128^3) ≈ 4 TB — check free space first.
# Run this before training with --dataset-name byu --base-data-dir "$PATCH_OUTPUT_DIR"
# =========================

set -euo pipefail

date
hostname

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cryoet

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit 1

# =========================
# Paths
# =========================
BYU_ROOT="/media/sumin/T9/byu-locating-bacterial-flagellar-motors-2025"
LABELS_CSV="${BYU_ROOT}/train_labels.csv"
IMAGES_DIR="${BYU_ROOT}/nifti/train"

DETECTION_BASE="/media/sumin/T9/byu_detection"
BYU_CSV="${DETECTION_BASE}/point_annotations.csv"
BYU_SPLIT_DATALIST="${DETECTION_BASE}/byu_split_datalist.json"

# loaders.py make_detection_dataset_3d reads "{base-data-dir}/{dataset_name}_100_datalist.json"
PATCH_OUTPUT_DIR="${DETECTION_BASE}/byu_detection_patches_128"
OUTPUT_JSON="${PATCH_OUTPUT_DIR}/byu_100_datalist.json"

# Gaussian target sigma (Å) — 3rd-place solution; sigma_vox = 200 / voxel_size
BYU_SIGMAS_ANG='{"motor":200}'

PATCH_SIZE=128
SEED=42

echo "=============================================="
echo "Step 1: BYU split + point annotations"
echo "  labels csv : $LABELS_CSV"
echo "  images dir : $IMAGES_DIR"
echo "  output dir : $DETECTION_BASE"
echo "=============================================="

python preprocessing/create_byu_detection_split.py \
    --labels-csv "$LABELS_CSV" \
    --images-dir "$IMAGES_DIR" \
    --output-dir "$DETECTION_BASE" \
    --train 70 --val 15 --test 15 \
    --seed "$SEED"

echo "=============================================="
echo "Step 2: Generating detection patches (BYU)"
echo "  split datalist : $BYU_SPLIT_DATALIST"
echo "  annotations csv: $BYU_CSV"
echo "  output dir     : $PATCH_OUTPUT_DIR"
echo "  output datalist: $OUTPUT_JSON"
echo "=============================================="

rm -rf "$PATCH_OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_JSON")"

python preprocessing/downstream_patch_generation.py \
    --detection \
    --zscore \
    --patch-size "$PATCH_SIZE" \
    --datalist-json "$BYU_SPLIT_DATALIST" \
    --csv           "$BYU_CSV" \
    --output-dir    "$PATCH_OUTPUT_DIR" \
    --output-json   "$OUTPUT_JSON" \
    --sigmas-ang    "$BYU_SIGMAS_ANG"

echo "Patchification complete."
echo "  Patches : $PATCH_OUTPUT_DIR"
echo "  Datalist: $OUTPUT_JSON"
date
