#!/bin/bash
# Inspect all detection datasets and save stats JSON to finetuning_detection dir
# Usage: bash slurm_scripts/inspect_detection_datasets.sh

source ~/.bashrc
conda activate cryoet

SCRIPT="/cluster/home/t139212uhn/scripts/cryoet/CryoET/dataset/downstream/inspect_dataset.py"
DETECTION_BASE="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream_detection"
OUTPUT_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/finetuning_detection"

mkdir -p "$OUTPUT_DIR"

# byu_dataset is itself a dataset (imagesTr/labelsTr directly inside)
BYU_DIR="${DETECTION_BASE}/byu_dataset"
echo "=============================================="
echo "Inspecting: byu_dataset"
echo "Output: ${OUTPUT_DIR}/byu_dataset_stats.json"
echo "=============================================="
python "$SCRIPT" "$BYU_DIR" --output "${OUTPUT_DIR}/byu_dataset_stats.json"

# czi_dataset contains Dataset* subdirectories
shopt -s nullglob
for DATASET_DIR in "${DETECTION_BASE}/czi_dataset/Dataset"*/; do
    DATASET_NAME="$(basename "$DATASET_DIR")"
    OUTPUT_JSON="${OUTPUT_DIR}/${DATASET_NAME}_stats.json"

    echo "=============================================="
    echo "Inspecting: $DATASET_NAME"
    echo "Output: $OUTPUT_JSON"
    echo "=============================================="

    python "$SCRIPT" "$DATASET_DIR" --output "$OUTPUT_JSON"
done
shopt -u nullglob

echo "Done!"
