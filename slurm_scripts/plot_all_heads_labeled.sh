#!/bin/bash
#SBATCH -J plot-heads-lbl
#SBATCH -p himem
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=slurm-%x-%j.out

source /cluster/home/t149164uhn/miniconda3/etc/profile.d/conda.sh
conda activate cryodino

D="/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets"
VIS_BASE="${HOME}/projects/cryoet_pca/vis_output_336"
FIG_BASE="${HOME}/projects/cryoet_pca/figures_labeled_336"
NAMES=("TS_0001" "TS_0009" "TS_027" "TS_100_3" "TS_100_4" "tomo_01a877")
LABELS=(
  "$D/downstream/Dataset001_CZII_10001/labelsTr/TS_0001.nii.gz"
  "$D/downstream/Dataset001_CZII_10001/labelsTr/TS_0009.nii.gz"
  "$D/downstream/Dataset000_CZII_10000/labelsTr/TS_027.nii.gz"
  "$D/downstream_detection/czi_dataset/Dataset445_CZII_10445/labelsTr/TS_100_3.nii.gz"
  "$D/downstream_detection/czi_dataset/Dataset446_CZII_10446/labelsTr/TS_100_4.nii.gz"
  "$D/downstream_detection/byu_dataset/labelsTr/tomo_01a877.nii.gz"
)

cd /cluster/home/t149164uhn/projects/CryoDINO
for MODEL in highres128 highres112; do
  mkdir -p "${FIG_BASE}/${MODEL}"
  for i in "${!NAMES[@]}"; do
    NAME="${NAMES[$i]}"; LBL="${LABELS[$i]}"
    echo "=== ${MODEL} ${NAME} | label: ${LBL} ==="
    if [[ ! -r "$LBL" ]]; then echo "LABEL NOT READABLE: $LBL"; continue; fi
    if [[ "$MODEL" == highres128 ]]; then HEAD=10; else HEAD=13; fi
    python visualization/plot_all_heads.py \
      --pca-dir  "${VIS_BASE}/${MODEL}/${NAME}/pca" \
      --mhsa-dir "${VIS_BASE}/${MODEL}/${NAME}/mhsa" \
      --output-dir "${FIG_BASE}/${MODEL}" \
      --name "$NAME" --lbl-path "$LBL" \
      --contrast-thr 0 --bg-thr 0.25 --bg-invert --head-idx "$HEAD" || echo "FAILED: ${MODEL} ${NAME}"
  done
done
echo "Done"
