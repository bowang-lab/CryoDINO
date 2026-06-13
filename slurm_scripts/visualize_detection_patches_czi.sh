#!/bin/bash
#SBATCH -J vis-detection-patches-czi
#SBATCH -p cpu_bwanggroup
#SBATCH -t 1:00:00
#SBATCH --account=bwanggroup_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=sum.kim@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/projects/bwanggroup/sumin/logs/%x_%j.log

# =========================
# Visualize CZI detection patches with overlaid point annotations.
# Run AFTER patchify_detection_czi.sh.
#
# Outputs: one PNG per patch (3 orthogonal mid-plane slices, dots per class).
# =========================

date
hostname
pwd

source ~/.bashrc
conda activate cryoet

cd /cluster/home/t129616uhn/projects/CryoDINO || exit 1

# =========================
# Paths (must match patchify_detection_czi.sh)
# =========================
#OUTPUT_JSON="/cluster/projects/bwanggroup/sumin/cryoet/experiments/czi_100_datalist.json"
#VIS_OUTPUT_DIR="/cluster/projects/bwanggroup/sumin/cryoet/czi_detection_patch_vis"
DETECTION_BASE="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream_detection"
PATCH_OUTPUT_DIR="${DETECTION_BASE}/Dataset440_CZII_10440_detection_patches_128"
#PATCH_OUTPUT_DIR="/cluster/projects/bwanggroup/sumin/Dataset440_CZII_10440_detection_patches_128"
# loaders.py make_detection_dataset_3d reads "{base-data-dir}/{dataset_name}_100_datalist.json"
#OUTPUT_JSON="/cluster/projects/bwanggroup/sumin/cryoet/experiments/czi_100_datalist.json"
OUTPUT_JSON="${PATCH_OUTPUT_DIR}/czi_100_datalist.json"
VIS_OUTPUT_DIR="${PATCH_OUTPUT_DIR}/Dataset440_CZII_10440_detection_patch_vis"

# =========================
# Options
# =========================
SPLIT="training"          # training | validation | test
N_PATCHES=-1              # how many patches to visualize; -1 = all
SLICE_TOL=5               # ±voxels around mid-plane to show a dot
SEED=42

echo "=============================================="
echo "Visualizing CZI detection patches"
echo "  datalist   : $OUTPUT_JSON"
echo "  output dir : $VIS_OUTPUT_DIR"
echo "  split      : $SPLIT"
echo "  n-patches  : $N_PATCHES"
echo "  slice-tol  : $SLICE_TOL"
echo "=============================================="

python visualization/visualize_detection_patches.py \
    --datalist      "$OUTPUT_JSON" \
    --output-dir    "$VIS_OUTPUT_DIR" \
    --split         "$SPLIT" \
    --n-patches     "$N_PATCHES" \
    --slice-tol     "$SLICE_TOL" \
    --only-particles \
    --seed          "$SEED"

echo "Visualization complete → $VIS_OUTPUT_DIR"
date
