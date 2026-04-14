#!/bin/bash
# Inspect all detection datasets and save stats JSON to finetuning_detection dir
# Usage: bash slurm_scripts/inspect_detection_datasets.sh

source ~/.bashrc
conda activate cryoet

SCRIPT="/cluster/home/t139212uhn/scripts/cryoet/CryoDINO/dataset/downstream/inspect_dataset.py"
DETECTION_BASE="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream_detection"
OUTPUT_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/finetuning_detection"

mkdir -p "$OUTPUT_DIR"

for SUBSET in byu_dataset czi_dataset; do
    SUBSET_DIR="${DETECTION_BASE}/${SUBSET}"
    if [ ! -d "$SUBSET_DIR" ]; then
        echo "[SKIP] $SUBSET_DIR not found"
        continue
    fi

    for DATASET_DIR in "$SUBSET_DIR"/Dataset*/; do
        DATASET_NAME="$(basename "$DATASET_DIR")"
        OUTPUT_JSON="${OUTPUT_DIR}/${DATASET_NAME}_stats.json"

        echo "=============================================="
        echo "Inspecting: $DATASET_NAME"
        echo "Output: $OUTPUT_JSON"
        echo "=============================================="

        python "$SCRIPT" "$DATASET_DIR" --output "$OUTPUT_JSON"
    done
done

echo "Done!"
