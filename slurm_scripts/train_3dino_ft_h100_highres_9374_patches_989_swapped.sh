#!/bin/bash
#SBATCH -J 3dino-ft-patches-pretrain-aug-989-swapped
#SBATCH -p gpu_bwanggroup
#SBATCH -t 6-00:00:00
#SBATCH --account=bwanggroup_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

# =========================
# Fine-tuning: Dataset989 with swapped train/val
# Train: 00004_0000  |  Val: 00012_0000  |  Test: 00011_0000
# Pretrain-matched augmentations, Frozen ViTAdapterUNETR
# =========================

date
hostname
pwd
nvidia-smi

source ~/.bashrc
conda activate cryoet

cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO || exit 1

# =========================
# Generate patches from training tomogram (00004_0000)
# =========================
RAW_DATALIST="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/Dataset989_EMPIAR_10989_transposed_swapped_100_datalist.json"
PATCH_OUTPUT_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream/Dataset989_EMPIAR_10989_transposed_swapped_train_patches_512"
PATCHES_DATALIST="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/Dataset989_EMPIAR_10989_transposed_swapped_patches512_100_datalist.json"

echo "=============================================="
echo "Generating patches from 00004_0000 (swapped train)..."
echo "=============================================="

python preprocessing/downstream_patch_generation.py \
    --zscore \
    --fg-threshold 0.0 \
    --patch-size 512 \
    --datalist-json "$RAW_DATALIST" \
    --output-dir "$PATCH_OUTPUT_DIR" \
    --output-json "$PATCHES_DATALIST"

echo "Patch generation complete: $PATCHES_DATALIST"

# =========================
# Training
# =========================
cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1

CONFIG_FILE="dinov2/configs/train/vit3d_highres.yaml"
PRETRAINED_WEIGHTS="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100_high_res/eval/training_9374/teacher_checkpoint.pth"
BASE_OUTPUT_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/finetuning"
BASE_DATA_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments"
DATASET_NAME="Dataset989_EMPIAR_10989_transposed_swapped_patches512"
SEGMENTATION_HEAD="ViTAdapterUNETR"
EPOCHS=100
EPOCH_LENGTH=300
EVAL_ITERS=600
WARMUP_ITERS=3000
IMAGE_SIZE=112
BATCH_SIZE=2
NUM_WORKERS=10
LEARNING_RATE=1e-4
DATASET_PERCENT=100
RESIZE_SCALE=1.0
OVERLAP=0.75
NUM_CLASSES=2

OUTPUT_DIR="${BASE_OUTPUT_DIR}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}_vit_adapter_pretrain_aug"
CACHE_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/cache_dir_downstream/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

echo "=============================================="
echo "Starting fine-tuning..."
echo "Dataset:   $DATASET_NAME"
echo "Output:    $OUTPUT_DIR"
echo "=============================================="

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

echo "Finished training: $OUTPUT_DIR"

# =========================
# Inference on test set (00011_0000)
# =========================
CHECKPOINT="${OUTPUT_DIR}/best_model.pth"
INFER_OUTPUT_DIR="${OUTPUT_DIR}/inference"
mkdir -p "$INFER_OUTPUT_DIR"

if [ ! -f "$CHECKPOINT" ]; then
    echo "  [SKIP inference] checkpoint not found: $CHECKPOINT"
else
    echo "----------------------------------------------"
    echo "Running inference..."
    echo "Checkpoint: $CHECKPOINT"
    echo "Output:     $INFER_OUTPUT_DIR"
    echo "----------------------------------------------"

    cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO || exit 1

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python inference/segmentation3d_inference.py \
      --config-file "3DINO/${CONFIG_FILE}" \
      --pretrained-weights "$PRETRAINED_WEIGHTS" \
      --checkpoint "$CHECKPOINT" \
      --segmentation-head "$SEGMENTATION_HEAD" \
      --image-size "$IMAGE_SIZE" \
      --num-classes "$NUM_CLASSES" \
      --datalist "$PATCHES_DATALIST" \
      --output-dir "$INFER_OUTPUT_DIR" \
      --dataset-name "Dataset989_EMPIAR_10989_transposed_swapped" \
      --overlap "$OVERLAP" \
      --batch-size "$BATCH_SIZE" \
      --cpu-metrics

    echo "Finished inference: $INFER_OUTPUT_DIR"
fi

echo "Done!"
date
