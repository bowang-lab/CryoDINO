#!/bin/bash
#SBATCH -J plot-heads-lbl
#SBATCH -p himem
#SBATCH -t 4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out

source /cluster/home/t149164uhn/miniconda3/etc/profile.d/conda.sh
conda activate cryodino

D="/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets"
VIS_BASE="${HOME}/projects/cryoet_pca/vis_output_336"
FIG_BASE="${HOME}/projects/cryoet_pca/figures_labeled_336"
NAMES=("TS_0001" "TS_0009" "TS_027" "TS_100_3" "TS_100_4" "tomo_01a877" "TS_028" "TS_034" "TS_037" "TS_041" "TS_0008")
IMAGES=(
    "/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/attention_maps/imgs/TS_0001_0000.nii.gz"
    "/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/attention_maps/imgs/TS_0009_0000.nii.gz"
    "/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/attention_maps/imgs/TS_027_0000.nii.gz"
    "$D/downstream_detection/czi_dataset/Dataset445_CZII_10445/imagesTr/TS_100_3_0000.nii.gz"
    "$D/downstream_detection/czi_dataset/Dataset446_CZII_10446/imagesTr/TS_100_4_0000.nii.gz"
    "$D/downstream_detection/train/tomo_01a877.nii.gz"
    "$D/downstream/Dataset000_CZII_10000/imagesTr/TS_028_0000.nii.gz"
    "$D/downstream/Dataset000_CZII_10000/imagesTr/TS_034_0000.nii.gz"
    "$D/downstream/Dataset000_CZII_10000/imagesTr/TS_037_0000.nii.gz"
    "$D/downstream/Dataset000_CZII_10000/imagesTr/TS_041_0000.nii.gz"
    "$D/downstream/Dataset001_CZII_10001/imagesTr/TS_0008_0000.nii.gz"
)

# Optional per-tomogram head per row (Axial Coronal Sagittal = rows A B C).
# Tomograms not listed here fall back to a single --head-idx "$HEAD" for every row.
declare -A ROW_HEADS=(
    ["TS_0008"]="8 12 0"
    ["TS_0001"]="0 0 12"
)

cd /cluster/home/t149164uhn/projects/CryoDINO
for MODEL in highres128 highres112; do
  mkdir -p "${FIG_BASE}/${MODEL}"
  for i in "${!NAMES[@]}"; do
    NAME="${NAMES[$i]}"; LBL="${LABELS[$i]}"
    echo "=== ${MODEL} ${NAME} | label: ${LBL} ==="
    if [[ ! -r "$LBL" ]]; then echo "LABEL NOT READABLE: $LBL"; continue; fi
    if [[ "$MODEL" == highres128 ]]; then HEAD=10; else HEAD=10; fi
    if [[ -n "${ROW_HEADS[$NAME]:-}" ]]; then
      HEAD_ARGS=(--row-heads ${ROW_HEADS[$NAME]})
    else
      HEAD_ARGS=(--head-idx "$HEAD")
    fi
    python visualization/plot_all_heads.py \
      --pca-dir  "${VIS_BASE}/${MODEL}/${NAME}/pca" \
      --mhsa-dir "${VIS_BASE}/${MODEL}/${NAME}/mhsa" \
      --output-dir "${FIG_BASE}/${MODEL}" \
      --name "$NAME" --lbl-path "$LBL" \
      --contrast-thr 0.2 --bg-thr 0.3 "${HEAD_ARGS[@]}" || echo "FAILED: ${MODEL} ${NAME}"
  done
done
echo "Done"
