#!/bin/bash
#SBATCH -J plot-pca-grid
#SBATCH -p himem
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out

# All-sample PCA overview (one row per tomogram; Orig|Label|PCA for A/C/S).
# TS_100_3 and TS_100_4 are excluded (bad PCA / near-empty labels).
#
# Usage (from ~/projects/CryoDINO):
#   sbatch slurm_scripts/plot_pca_grid.sh
#   MODEL=highres112 BG_THR=0.25 EXTRA="--bg-invert" sbatch slurm_scripts/plot_pca_grid.sh

source /cluster/home/t149164uhn/miniconda3/etc/profile.d/conda.sh
conda activate cryodino

MODEL="${MODEL:-highres128}"
BG_THR="${BG_THR:-0.3}"
CONTRAST_THR="${CONTRAST_THR:-0.2}"
EXTRA="${EXTRA:-}"

D="/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets"
VIS_BASE="${HOME}/projects/cryoet_pca/vis_output_336/${MODEL}"
OUT="${HOME}/projects/cryoet_pca/figures_pca_grid/${MODEL}_pca_grid.png"

SAMPLES=(TS_0001 TS_0009 TS_0008 TS_027 TS_028 TS_034 TS_037 TS_041 tomo_01a877)
L0="$D/downstream/Dataset000_CZII_10000/labelsTr"
L1="$D/downstream/Dataset001_CZII_10001/labelsTr"
LABEL_ARGS=(
  --label "TS_0001=$L1/TS_0001.nii.gz"
  --label "TS_0009=$L1/TS_0009.nii.gz"
  --label "TS_0008=$L1/TS_0008.nii.gz"
  --label "TS_027=$L0/TS_027.nii.gz"
  --label "TS_028=$L0/TS_028.nii.gz"
  --label "TS_034=$L0/TS_034.nii.gz"
  --label "TS_037=$L0/TS_037.nii.gz"
  --label "TS_041=$L0/TS_041.nii.gz"
)
# Per-sample background threshold overrides, e.g. ("--bg-thr-for" "TS_0009=0.25")
BG_OVERRIDES=()

cd /cluster/home/t149164uhn/projects/CryoDINO
python visualization/plot_pca_grid.py \
  --vis-base "$VIS_BASE" \
  --samples "${SAMPLES[@]}" \
  "${LABEL_ARGS[@]}" \
  --output "$OUT" \
  --contrast-thr "$CONTRAST_THR" --bg-thr "$BG_THR" \
  "${BG_OVERRIDES[@]}" $EXTRA
