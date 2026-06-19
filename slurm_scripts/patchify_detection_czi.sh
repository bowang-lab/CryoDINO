#!/bin/bash
#SBATCH -J patchify-detection-czi
#SBATCH -p gpu_bwanggroup
#SBATCH -t 4:00:00
#SBATCH --account=bwanggroup_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --mail-user=sum.kim@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/projects/bwanggroup/sumin/logs/%x_%j.log

# =========================
# Detection patch generation (CZI)
# Runs downstream_patch_generation.py in --detection mode:
#   CSV point annotations → overlapping 128^3 .pt patches + czi_100_datalist.json
# Run this before train_3dino_ft_h100_detection_czi.sh
# =========================

date
hostname
pwd

source ~/.bashrc
conda activate cryoet

cd /cluster/home/t129616uhn/projects/CryoDINO || exit 1

# =========================
# Paths
# =========================
DETECTION_BASE="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream_detection"
BASE_DATA_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments"

# Raw CZI detection dataset (image NIfTIs + CSV point annotations + split datalist)
CZI_SPLIT_DATALIST="${DETECTION_BASE}/czi_dataset/Dataset440_CZII_10440/czi_split_datalist.json"
CZI_CSV="${DETECTION_BASE}/czi_dataset/Dataset440_CZII_10440/point_annotations.csv"

# Detection patch output (overlapping 128^3 patches + datalist named for loaders.py)
PATCH_OUTPUT_DIR="${DETECTION_BASE}/Dataset440_CZII_10440_detection_patches_128"
#PATCH_OUTPUT_DIR="/cluster/projects/bwanggroup/sumin/Dataset440_CZII_10440_detection_patches_128"
# loaders.py make_detection_dataset_3d reads "{base-data-dir}/{dataset_name}_100_datalist.json"
#OUTPUT_JSON="/cluster/projects/bwanggroup/sumin/cryoet/experiments/czi_100_datalist.json"
OUTPUT_JSON="${PATCH_OUTPUT_DIR}/czi_100_datalist.json"

# CZI 6-class sigma map (particle radius in Angstroms → sigma_vox = radius / voxel_size)
CZI_SIGMAS_ANG='{"Beta-amylase":65,"Beta-galactosidase":90,"Thyroglobulin":130,"cytosolic ribosome":150,"ferritin complex":60,"virus-like capsid":135}'

PATCH_SIZE=128

echo "=============================================="
echo "Generating detection patches (CZI)"
echo "  split datalist : $CZI_SPLIT_DATALIST"
echo "  annotations csv: $CZI_CSV"
echo "  output dir     : $PATCH_OUTPUT_DIR"
echo "  output datalist: $OUTPUT_JSON"
echo "=============================================="

rm -rf "$PATCH_OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_JSON")"

python preprocessing/downstream_patch_generation.py \
    --detection \
    --zscore \
    --patch-size "$PATCH_SIZE" \
    --datalist-json "$CZI_SPLIT_DATALIST" \
    --csv           "$CZI_CSV" \
    --output-dir    "$PATCH_OUTPUT_DIR" \
    --output-json   "$OUTPUT_JSON" \
    --sigmas-ang    "$CZI_SIGMAS_ANG"

echo "Patchification complete."
echo "  Patches : $PATCH_OUTPUT_DIR"
echo "  Datalist: $OUTPUT_JSON"
date
