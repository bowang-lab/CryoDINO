#!/bin/bash
#SBATCH -J patchify-detection-byu
#SBATCH -p all
#SBATCH -t 5-00:00:00
#SBATCH --account=pmccaiteam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=sum.kim@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/projects/bwanggroup/sumin/logs/%x_%j.log

# =========================
# Detection patch generation (BYU flagellar motors) — cluster
#   Step 1: train_labels.csv → point_annotations.csv + motor-count-stratified
#           70/15/15 tomogram-level split (byu_split_datalist.json)
#           (aborts if any imagesTr NIfTI shape != CSV array shape in X,Y,Z order)
#   Step 2: downstream_patch_generation.py --detection
#           → overlapping 128^3 .pt patches (train) + byu_100_datalist.json
#   Step 3: create_detection_percent_datalists.py → nested tomogram-level 10% / 50%
#           training subsets in byu_{10,50}percent/byu_100_datalist.json
# Native voxel spacing (6.5–19.7 Å/vox); sigma converted per tomogram (200 Å / spacing).
# Patches stored as raw uint8 (--native-dtype, no z-score): train-time per-crop
#   percentile rescale makes the model input identical to z-scored float32 patches.
# Disk: 533,162 train patches × 2.1 MB (uint8 128^3) ≈ 1.1 TB.
# Run this before training with --dataset-name byu --base-data-dir "$PATCH_OUTPUT_DIR"
#   (or "$PATCH_OUTPUT_DIR/byu_10percent", "$PATCH_OUTPUT_DIR/byu_50percent")
# =========================

date
hostname

# /etc/bashrc reads unset variables, so enable strict mode after conda activate
source ~/.bashrc
conda activate cryoet
set -euo pipefail

cd /cluster/home/t129616uhn/projects/CryoDINO || exit 1

# =========================
# Paths
# =========================
BYU_ROOT="/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets/downstream_detection/byu_dataset"
LABELS_CSV="${BYU_ROOT}/train_labels.csv"
IMAGES_DIR="/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets/downstream_detection/train"

BYU_CSV="${BYU_ROOT}/point_annotations.csv"
BYU_SPLIT_DATALIST="${BYU_ROOT}/byu_split_datalist.json"

# loaders.py make_detection_dataset_3d reads "{base-data-dir}/{dataset_name}_100_datalist.json"
PATCH_OUTPUT_DIR="/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets/downstream_detection/byu_detection_patches_128"
OUTPUT_JSON="${PATCH_OUTPUT_DIR}/byu_100_datalist.json"

# Gaussian target sigma (Å) — 3rd-place solution; sigma_vox = 200 / voxel_size
BYU_SIGMAS_ANG='{"motor":200}'

PATCH_SIZE=128
SEED=42

echo "=============================================="
echo "Step 1: BYU split + point annotations"
echo "  labels csv : $LABELS_CSV"
echo "  images dir : $IMAGES_DIR"
echo "  output dir : $BYU_ROOT"
echo "=============================================="

# Reuse an existing split so re-runs never change it; delete both files to regenerate
if [[ -f "$BYU_SPLIT_DATALIST" && -f "$BYU_CSV" ]]; then
    echo "Split already exists — skipping Step 1"
else
    python -u preprocessing/create_byu_detection_split.py \
        --labels-csv "$LABELS_CSV" \
        --images-dir "$IMAGES_DIR" \
        --output-dir "$BYU_ROOT" \
        --train 70 --val 15 --test 15 \
        --seed "$SEED"
fi

echo "=============================================="
echo "Step 2: Generating detection patches (BYU)"
echo "  split datalist : $BYU_SPLIT_DATALIST"
echo "  annotations csv: $BYU_CSV"
echo "  output dir     : $PATCH_OUTPUT_DIR"
echo "  output datalist: $OUTPUT_JSON"
echo "=============================================="

rm -rf "$PATCH_OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_JSON")"

python -u preprocessing/downstream_patch_generation.py \
    --detection \
    --native-dtype \
    --patch-size "$PATCH_SIZE" \
    --datalist-json "$BYU_SPLIT_DATALIST" \
    --csv           "$BYU_CSV" \
    --output-dir    "$PATCH_OUTPUT_DIR" \
    --output-json   "$OUTPUT_JSON" \
    --sigmas-ang    "$BYU_SIGMAS_ANG"

echo "=============================================="
echo "Step 3: 10% / 50% tomogram-level training subsets"
echo "=============================================="

python -u preprocessing/create_detection_percent_datalists.py \
    --dataset-name byu \
    --split-datalist "$BYU_SPLIT_DATALIST" \
    --patch-datalist "$OUTPUT_JSON" \
    --percents 10 50 \
    --seed "$SEED"

echo "Patchification complete."
echo "  Patches : $PATCH_OUTPUT_DIR"
echo "  Datalist: $OUTPUT_JSON"
echo "  Subsets : ${PATCH_OUTPUT_DIR}/byu_{10,50}percent/byu_100_datalist.json"
date
