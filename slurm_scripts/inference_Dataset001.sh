#!/bin/bash
#SBATCH -J 3dino-inference-ds001
#SBATCH -p gpu_bwanggroup
#SBATCH -t 1-12:00:00
#SBATCH --account=bwanggroup_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=220G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

# =========================
# Inference: Dataset001_CZII_10001 (4-class)
# Covers all 8 variants: UNETR/ViTAdapter × 100%/10%/50%/full_ft
# =========================

date; hostname; pwd; nvidia-smi

source ~/.bashrc
conda activate cryoet
cd /cluster/home/t139212uhn/scripts/cryoet/CryoET || exit 1

CONFIG_FILE="3DINO/dinov2/configs/train/vit3d_highres.yaml"
PRETRAINED_WEIGHTS="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100_high_res/eval/training_9374/teacher_checkpoint.pth"
FINETUNING_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/finetuning"
DATALIST_FILE="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/Dataset001_CZII_10001_patches512_100_datalist.json"
DATASET_NAME="Dataset001_CZII_10001"
NUM_CLASSES=4
IMAGE_SIZE=112
BATCH_SIZE=1
OVERLAP=0.75

EXPERIMENTS=(
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512"
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512_vit_adapter"
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512_10percent"
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512_10percent_vit_adapter"
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512_50percent"
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512_50percent_vit_adapter"
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512_full_ft"
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512_full_ft_vit_adapter"
)

SKIPPED=(); FAILED=()

for EXP in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Experiment: $EXP"
    echo "=============================================="

    CHECKPOINT="${FINETUNING_DIR}/${EXP}/best_model.pth"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "  [SKIP] checkpoint not found: $CHECKPOINT"
        SKIPPED+=("$EXP"); continue
    fi

    if [[ "$EXP" == *"_vit_adapter"* ]]; then SEG_HEAD="ViTAdapterUNETR"; else SEG_HEAD="UNETR"; fi

    OUTPUT_DIR="${FINETUNING_DIR}/${EXP}/inference"
    mkdir -p "$OUTPUT_DIR"

    echo "  Head: $SEG_HEAD  |  Checkpoint: $CHECKPOINT"

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python inference/segmentation3d_inference.py \
        --config-file "$CONFIG_FILE" \
        --pretrained-weights "$PRETRAINED_WEIGHTS" \
        --checkpoint "$CHECKPOINT" \
        --segmentation-head "$SEG_HEAD" \
        --image-size "$IMAGE_SIZE" \
        --num-classes "$NUM_CLASSES" \
        --datalist "$DATALIST_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --dataset-name "$DATASET_NAME" \
        --overlap "$OVERLAP" \
        --batch-size "$BATCH_SIZE" \
        --cpu-metrics

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then echo "  [FAILED] exit $EXIT_CODE"; FAILED+=("$EXP")
    else echo "  [DONE]"; fi
done

echo ""
echo "====== SUMMARY ======"
echo "Skipped: ${#SKIPPED[@]}"; for s in "${SKIPPED[@]}"; do echo "  - $s"; done
echo "Failed:  ${#FAILED[@]}";  for f in "${FAILED[@]}";  do echo "  - $f"; done
echo "Done: $((${#EXPERIMENTS[@]} - ${#SKIPPED[@]} - ${#FAILED[@]})) / ${#EXPERIMENTS[@]}"
date
