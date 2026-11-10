#!/bin/bash
#SBATCH -J 3dino-vis-attn-maps
#SBATCH -p gpu_pmccaiteam
#SBATCH -t 1-00:00:00
#SBATCH --account=pmccaiteam_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --output=slurm-%x-%j.out

set -eo pipefail

date
hostname
nvidia-smi

source /cluster/home/t149164uhn/miniconda3/etc/profile.d/conda.sh
conda activate cryodino

REPO_ROOT="/cluster/home/t149164uhn/projects/CryoDINO"
DINOV2_ROOT="${REPO_ROOT}/3DINO"

MODEL="${MODEL:?Submit with MODEL=highres128 or MODEL=highres112}"
SAMPLE_FILTER="${SAMPLE_FILTER:-all}"
VIS_TYPES="${VIS_TYPES:-pca mhsa}"
OUT_BASE="${OUT_BASE:-${HOME}/projects/cryoet_pca/vis_output_v2}"

if [[ ! "$VIS_TYPES" =~ ^[[:space:]]*(pca|mhsa)?([[:space:]]+(pca|mhsa))*[[:space:]]*$ ]]; then
    echo "Unsupported VIS_TYPES='$VIS_TYPES'; use pca and/or mhsa." >&2
    exit 2
fi

case "$MODEL" in
    highres128)
        CONFIG="dinov2/configs/train/vit3d_highres.yaml"
        WEIGHTS="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/ssl3d_run_b200_high_res_128/eval/training_6249/teacher_checkpoint.pth"
        IMAGE_SIZE=128
        ;;
    highres112)
        CONFIG="dinov2/configs/train/vit3d_highres_112.yaml"
        WEIGHTS="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/ssl3d_run_b200_high_res/eval/training_6249/teacher_checkpoint.pth"
        IMAGE_SIZE=112
        ;;
    *)
        echo "Unsupported MODEL='$MODEL'; use highres128 or highres112." >&2
        exit 2
        ;;
esac

INPUT_TYPE="${INPUT_TYPE:-sliding_window}"
RENDER_SIZE="${RENDER_SIZE:-$IMAGE_SIZE}"

CONFIG_PATH="${DINOV2_ROOT}/${CONFIG}"
VIS_SCRIPT="${DINOV2_ROOT}/dinov2/eval/vis_pca_cryoet.py"

if [[ ! -r "$CONFIG_PATH" ]]; then
    echo "Config is not readable: $CONFIG_PATH" >&2
    exit 1
fi
if [[ ! -r "$WEIGHTS" ]]; then
    echo "Checkpoint is not readable: $WEIGHTS" >&2
    exit 1
fi

D="/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets"
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

if [[ ! -r "$VIS_SCRIPT" ]]; then
    echo "Visualization script is not readable: $VIS_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUT_BASE"
cd "$DINOV2_ROOT"
matched_sample=false

echo "VIS_TYPES: ${VIS_TYPES}"
echo "OUT_BASE: ${OUT_BASE}"

for i in "${!NAMES[@]}"; do
    NAME="${NAMES[$i]}"
    IMG="${IMAGES[$i]}"

    if [[ "$SAMPLE_FILTER" != all && "$SAMPLE_FILTER" != "$NAME" ]]; then
        continue
    fi
    matched_sample=true

    if [[ ! -r "$IMG" ]]; then
        echo "Image is not readable: $IMG" >&2
        exit 1
    fi

    for VIS_TYPE in $VIS_TYPES; do
        OUT_DIR="${OUT_BASE}/${MODEL}/${NAME}/${VIS_TYPE}"
        mkdir -p "$OUT_DIR"

        echo "=============================================="
        echo "Model: ${MODEL} | Sample: ${NAME} | Vis type: ${VIS_TYPE}"
        echo "VIS_TYPES: ${VIS_TYPES}"
        echo "OUT_BASE: ${OUT_BASE}"
        echo "Image: ${IMG}"
        echo "Output: ${OUT_DIR}"
        echo "=============================================="

        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        PYTHONPATH=. python "$VIS_SCRIPT" \
            --config-file "$CONFIG" \
            --pretrained-weights "$WEIGHTS" \
            --image-path "$IMG" \
            --output-dir "$OUT_DIR" \
            --vis-type "$VIS_TYPE" \
            --input-type "$INPUT_TYPE" \
            --image-size "$RENDER_SIZE"
    done
done

if [[ "$matched_sample" != true ]]; then
    echo "Unknown SAMPLE_FILTER='$SAMPLE_FILTER'." >&2
    echo "Choose all or one of: ${NAMES[*]}" >&2
    exit 2
fi

echo "All requested visualizations complete."
date
