#!/bin/bash
#SBATCH -J plot-row-heads
#SBATCH -p himem
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out

# One figure per tomogram, with a different attention head per row
# (rows: Axial, Coronal, Sagittal).
#
# Usage (from ~/projects/CryoDINO):
#   sbatch slurm_scripts/plot_row_heads.sh
#   MODEL=highres112 BG_THR=0.25 EXTRA="--bg-invert" sbatch slurm_scripts/plot_row_heads.sh

source /cluster/home/t149164uhn/miniconda3/etc/profile.d/conda.sh
conda activate cryodino

MODEL="${MODEL:-highres128}"
BG_THR="${BG_THR:-0.3}"
CONTRAST_THR="${CONTRAST_THR:-0.2}"
EXTRA="${EXTRA:-}"

D="/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets"
VIS_BASE="${HOME}/projects/cryoet_pca/vis_output_336"
FIG_BASE="${HOME}/projects/cryoet_pca/figures_row_heads_336/${MODEL}"
mkdir -p "$FIG_BASE"

# name -> "axial coronal sagittal" heads
declare -A ROW_HEADS=(
    ["TS_0008"]="8 12 0"
    ["TS_0001"]="0 0 12"
)
declare -A LABELS=(
    ["TS_0008"]="$D/downstream/Dataset001_CZII_10001/labelsTr/TS_0008.nii.gz"
    ["TS_0001"]="$D/downstream/Dataset001_CZII_10001/labelsTr/TS_0001.nii.gz"
)

cd /cluster/home/t149164uhn/projects/CryoDINO
for NAME in "${!ROW_HEADS[@]}"; do
  LBL="${LABELS[$NAME]}"
  PCA_DIR="${VIS_BASE}/${MODEL}/${NAME}/pca"
  MHSA_DIR="${VIS_BASE}/${MODEL}/${NAME}/mhsa"
  echo "=== ${MODEL} ${NAME} | heads: ${ROW_HEADS[$NAME]} | label: ${LBL} ==="
  if [[ ! -d "$PCA_DIR" || ! -d "$MHSA_DIR" ]]; then
    echo "MISSING VIS OUTPUT: $PCA_DIR or $MHSA_DIR (run vis_attention_maps.sh first)"; continue
  fi
  if [[ ! -r "$LBL" ]]; then echo "LABEL NOT READABLE: $LBL"; continue; fi
  python visualization/plot_all_heads.py \
    --pca-dir "$PCA_DIR" --mhsa-dir "$MHSA_DIR" \
    --output-dir "$FIG_BASE" \
    --name "${NAME}_rowheads" --lbl-path "$LBL" \
    --contrast-thr "$CONTRAST_THR" --bg-thr "$BG_THR" \
    --row-heads ${ROW_HEADS[$NAME]} $EXTRA || echo "FAILED: ${MODEL} ${NAME}"
done
echo "Done"
