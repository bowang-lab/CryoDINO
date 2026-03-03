#!/bin/bash
#SBATCH -J 3dino-lp-job2
#SBATCH -p gpu_bwanggroup
#SBATCH -t 4-00:00:00
#SBATCH --account=bwanggroup_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=110G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

# =========================
# Job 2: h100 (37499, 24999, 12499) + h100_high_res (all 4)
# =========================

date
hostname
pwd
nvidia-smi

source ~/.bashrc
conda activate cryoet

cd /cluster/home/t139212uhn/scripts/cryoet/CryoET/3DINO || exit 1

pretrained_weights_paths=(
    "/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100/eval/training_37499/teacher_checkpoint.pth"
    "/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100/eval/training_24999/teacher_checkpoint.pth"
    "/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100/eval/training_12499/teacher_checkpoint.pth"
    "/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100_high_res/eval/training_3124/teacher_checkpoint.pth"
    "/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100_high_res/eval/training_6249/teacher_checkpoint.pth"
    "/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100_high_res/eval/training_9374/teacher_checkpoint.pth"
    "/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100_high_res/eval/training_12499/teacher_checkpoint.pth"
)

downstream_datasets=(
    "Dataset001_CZII_10001_patches512"
    "Dataset010_CZII_10010_patches512"
    "Dataset989_EMPIAR_10989_transposed_patches512"
)

# =========================
# Fixed Parameters
# =========================
CONFIG_DEFAULT="dinov2/configs/ssl3d_default_config.yaml"
CONFIG_HIGHRES="dinov2/configs/train/vit3d_highres.yaml"
BASE_OUTPUT_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/linear_probing"
DATASET_PERCENT=100
BASE_DATA_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments"
SEGMENTATION_HEAD="Linear"
EPOCHS=100
EPOCH_LENGTH=125
EVAL_ITERS=600
WARMUP_ITERS=3000
BATCH_SIZE=2
NUM_WORKERS=7
LEARNING_RATE=0.001
CACHE_DIR_BASE="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/cache_dir_downstream"
RESIZE_SCALE=1.0

mkdir -p "$CACHE_DIR_BASE"

for PRETRAINED_WEIGHTS in "${pretrained_weights_paths[@]}"; do
    # Select config and image size based on high_res in path
    if [[ "$PRETRAINED_WEIGHTS" == *"high_res"* ]]; then
        CONFIG_FILE="$CONFIG_HIGHRES"
        IMAGE_SIZE=112
    else
        CONFIG_FILE="$CONFIG_DEFAULT"
        IMAGE_SIZE=96
    fi

    # Extract run name and iteration from path
    RUN_NAME=$(basename $(dirname $(dirname $(dirname "$PRETRAINED_WEIGHTS"))))
    ITERATION=$(basename $(dirname "$PRETRAINED_WEIGHTS"))

    for DATASET_NAME in "${downstream_datasets[@]}"; do
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}_${ITERATION}_${DATASET_NAME}"
        CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}"
        mkdir -p "$CACHE_DIR"

        echo "=============================================="
        echo "Starting 3D segmentation training..."
        echo "Config: $CONFIG_FILE"
        echo "Pretrained weights: $PRETRAINED_WEIGHTS"
        echo "Dataset: $DATASET_NAME"
        echo "Output directory: $OUTPUT_DIR"
        echo "=============================================="

        mkdir -p "$OUTPUT_DIR"

        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python dinov2/eval/segmentation3d.py \
          --config-file "$CONFIG_FILE" \
          --output-dir "$OUTPUT_DIR" \
          --pretrained-weights "$PRETRAINED_WEIGHTS" \
          --dataset-name "$DATASET_NAME" \
          --dataset-percent "$DATASET_PERCENT" \
          --base-data-dir "$BASE_DATA_DIR" \
          --segmentation-head "$SEGMENTATION_HEAD" \
          --epochs "$EPOCHS" \
          --epoch-length "$EPOCH_LENGTH" \
          --eval-iters "$EVAL_ITERS" \
          --warmup-iters "$WARMUP_ITERS" \
          --image-size "$IMAGE_SIZE" \
          --batch-size "$BATCH_SIZE" \
          --num-workers "$NUM_WORKERS" \
          --learning-rate "$LEARNING_RATE" \
          --cache-dir "$CACHE_DIR" \
          --resize-scale "$RESIZE_SCALE"

        echo "Finished: $OUTPUT_DIR"
        echo ""
    done
done

echo "Job 2 completed!"
date
