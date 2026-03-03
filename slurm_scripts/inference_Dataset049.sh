#!/bin/bash
#SBATCH -J 3dino-inference-ds049
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
# Inference: Dataset049_EMPIAR_12049 — two ViTAdapter experiments
#
# vit_adapter2: 4-class (classes 1,5 merged into bg; 2→1, 3→2, 4→3)
#               dataset_name contains "12049" → label remap applied in build_transforms
#
# vit_adapter:  6-class (original labels, trained before remap was introduced)
#               dataset_name avoids "12049" → no remap, raw 0-5 labels used
# =========================

date; hostname; pwd; nvidia-smi

source ~/.bashrc
conda activate cryoet
cd /cluster/home/t139212uhn/scripts/cryoet/CryoET || exit 1

CONFIG_FILE="3DINO/dinov2/configs/train/vit3d_highres.yaml"
PRETRAINED_WEIGHTS="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100_high_res/eval/training_9374/teacher_checkpoint.pth"
FINETUNING_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/finetuning"
DATALIST_FILE="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/Dataset049_EMPIAR_12049_transposed_patches512_100_datalist.json"
IMAGE_SIZE=112
BATCH_SIZE=2
OVERLAP=0.75

# Parallel arrays: experiment, num_classes, dataset_name (controls label transform)
EXPERIMENTS=(
    "ssl3d_run_h100_high_res_training_9374_Dataset049_EMPIAR_12049_transposed_patches512_vit_adapter2"
    "ssl3d_run_h100_high_res_training_9374_Dataset049_EMPIAR_12049_transposed_patches512_vit_adapter"
)
NUM_CLASSES_LIST=(
    4    # vit_adapter2: remapped to 4 classes
    6    # vit_adapter:  original 6-class labels (trained before remap)
)
DATASET_NAMES=(
    "Dataset049_EMPIAR_12049_transposed"   # contains "12049" → remap applied in build_transforms
    "Dataset049_EMPIAR_6class"             # avoids "12049" → no remap, raw 0-5 labels
)

SKIPPED=(); FAILED=()

for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    NUM_CLASSES="${NUM_CLASSES_LIST[$i]}"
    DATASET_NAME="${DATASET_NAMES[$i]}"

    echo ""
    echo "=============================================="
    echo "Experiment: $EXP"
    echo "  Classes: $NUM_CLASSES  |  Dataset name: $DATASET_NAME"
    echo "=============================================="

    CHECKPOINT="${FINETUNING_DIR}/${EXP}/best_model.pth"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "  [SKIP] checkpoint not found: $CHECKPOINT"
        SKIPPED+=("$EXP"); continue
    fi

    OUTPUT_DIR="${FINETUNING_DIR}/${EXP}/inference"
    mkdir -p "$OUTPUT_DIR"

    echo "  Checkpoint: $CHECKPOINT"
    echo "  Output:     $OUTPUT_DIR"

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python inference/segmentation3d_inference.py \
        --config-file "$CONFIG_FILE" \
        --pretrained-weights "$PRETRAINED_WEIGHTS" \
        --checkpoint "$CHECKPOINT" \
        --segmentation-head "ViTAdapterUNETR" \
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
