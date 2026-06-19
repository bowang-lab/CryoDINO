#!/bin/bash
#SBATCH -J 3dino-ft-detection-czi
#SBATCH -p gpu_bwanggroup
#SBATCH -t 3-00:00:00
#SBATCH --account=bwanggroup_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --mail-user=sum.kim@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t129616uhn/projects/logs/%x_%j.log

# =========================
# Fine-tuning: h100_high_res training_9374 — DETECTION (CZI)
# Pre-extracted 128^3 overlapping patches (.pt) with GT point coords, ViTAdapterUNETR head.
# Adapted from train_3dino_ft_h100_highres_9374_patches.sh (segmentation).
#
# Key differences vs segmentation:
#   * downstream_patch_generation.py runs in --detection mode (CSV point
#     annotations → "points" key, overlapping sliding-window patches).
#   * No mix-patch augmentation (that path is label-mask specific).
#   * dataset-name MUST be "czi" (loaders.py hardcodes czi=6 classes / byu=1);
#     base-data-dir must contain "czi_100_datalist.json".
#   * num-classes is derived inside loaders.py — NOT passed on the CLI.
#   * image-size must equal the patch size (128): detection make_transforms
#     ignores crop and the head uses image-size as the spatial dim.
#   * No separate inference call — detection3d.py runs its own test F4
#     threshold-sweep at the end and writes results.json.
# =========================

date
hostname
pwd
nvidia-smi

source ~/.bashrc
conda activate cryoet

cd /cluster/home/t129616uhn/projects/CryoDINO/3DINO || exit 1

# =========================
# Paths
# =========================
BASE_DATA_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream_detection/Dataset440_CZII_10440_detection_patches_128"
BASE_OUTPUT_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/finetuning_detection"
CACHE_DIR_BASE="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/cache_dir_downstream_detection"

# loaders.py make_detection_dataset_3d reads "{base-data-dir}/{dataset_name}_100_datalist.json"
OUTPUT_JSON="${BASE_DATA_DIR}/czi_100_datalist.json"

mkdir -p "$BASE_OUTPUT_DIR"

#cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1

# =========================
# Fixed Parameters
# =========================
CONFIG_FILE="dinov2/configs/train/vit3d_highres.yaml"
PRETRAINED_WEIGHTS="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100_high_res/eval/training_9374/teacher_checkpoint.pth"
DATASET_NAME="czi"            # loaders.py: czi → 6 classes (byu → 1)
DATASET_PERCENT=100
SEGMENTATION_HEAD="ViTAdapterUNETR"
EPOCHS=100
EPOCH_LENGTH=300
EVAL_ITERS=1500
WARMUP_ITERS=3000
IMAGE_SIZE=128                # must match PATCH_SIZE for detection
BATCH_SIZE=4                  # 128^3 patches are smaller than seg 512^3
NUM_WORKERS=10
LEARNING_RATE=1e-4
RESIZE_SCALE=1.0

OUTPUT_DIR="${BASE_OUTPUT_DIR}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}_detection_vit_adapter"
CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}_detection"

rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Starting 3D detection fine-tuning..."
echo "Config: $CONFIG_FILE"
echo "Pretrained weights: $PRETRAINED_WEIGHTS"
echo "Dataset: $DATASET_NAME  (datalist: $OUTPUT_JSON)"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python dinov2/eval/detection3d.py \
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

echo "Finished detection training: $OUTPUT_DIR"
echo "  (test F4 threshold-sweep + results.json written by detection3d.py)"
date
