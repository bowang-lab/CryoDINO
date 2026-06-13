#!/bin/bash
#SBATCH -J vis-detection-tomo-overlay-czi
#SBATCH -p cpu_bwanggroup
#SBATCH -t 1:00:00
#SBATCH --account=bwanggroup_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=sum.kim@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/projects/bwanggroup/sumin/logs/%x_%j.log

# =========================
# Visualize raw CZI tomograms (imagesTr) with overlaid point annotation blobs.
# Run BEFORE patchify_detection_czi.sh to sanity-check tomograms + annotations.
#
# Outputs per tomogram:
#   {run}_ortho.png  — XY / XZ / YZ mid-plane views with blob circles
#   {run}_mosaic.png — 4×4 Z-slice mosaic with annotation dots
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

IMAGES_DIR="${DETECTION_BASE}/czi_dataset/Dataset445_CZII_10445/imagesTr"
CZI_CSV="${DETECTION_BASE}/czi_dataset/Dataset445_CZII_10445/point_annotations.csv"

VIS_OUTPUT_DIR="${DETECTION_BASE}/Dataset445_CZII_10445_tomo_overlay_vis"

# =========================
# Options
# =========================
# Particle radii in Angstroms (used to size blob circles)
CZI_SIGMAS_ANG='{"Beta-amylase":65,"Beta-galactosidase":90,"Thyroglobulin":130,"cytosolic ribosome":150,"ferritin complex":60,"virus-like capsid":135}'

SLICE_TOL=3   # ±voxels around the slice plane to show a blob
N_TOMOS=-1    # -1 = all tomograms

echo "=============================================="
echo "Visualizing CZI tomograms with annotation blobs"
echo "  images dir : $IMAGES_DIR"
echo "  csv        : $CZI_CSV"
echo "  output dir : $VIS_OUTPUT_DIR"
echo "  slice-tol  : $SLICE_TOL"
echo "  n-tomos    : $N_TOMOS"
echo "=============================================="

python visualization/detection_tomogram_overlay.py \
    --images-dir  "$IMAGES_DIR" \
    --csv         "$CZI_CSV" \
    --output-dir  "$VIS_OUTPUT_DIR" \
    --sigmas-ang  "$CZI_SIGMAS_ANG" \
    --slice-tol   "$SLICE_TOL" \
    --n-tomos     "$N_TOMOS"

echo "Visualization complete → $VIS_OUTPUT_DIR"
date
