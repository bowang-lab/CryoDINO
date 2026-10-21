#!/bin/bash
#SBATCH -J cryodino-detection
#SBATCH -p gpu_pmccaiteam
#SBATCH -t 3-00:00:00
#SBATCH --account=pmccaiteam_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --output=/cluster/home/t129616uhn/projects/logs/%x_%j.log

# =========================
# Detection benchmark: one (method, dataset, percent) run of dinov2/eval/detection3d.py.
# Submit through submit_detection.sh, which sets these via --export:
#   METHOD  : cryodino | random_init | resnetfpn
#   DATASET : czi | byu
#   PERCENT : 100 | 50 | 10
#
# Methods (all three train under the same loss / assigner / NMS decode in detection3d.py):
#   cryodino    ViTAdapterUNETR on the B200 highres128 training_6249 backbone, backbone frozen
#               (same as the segmentation fine-tuning scripts, which don't pass --train-feature-model)
#   random_init ViTAdapterUNETR, no weights (empty path -> build_model_for_eval falls back to
#               random init), backbone trained (--train-feature-model, as in
#               slurm_scripts/train_3dino_ft_random_init_patches.sh)
#   resnetfpn   ResNet34-FPN from scratch; --pretrained-weights is ignored for this head
#
# Percent is selected by directory, not by --dataset-percent: each {czi,byu}_{10,50}percent/
# holds a pre-subsampled {dataset}_100_datalist.json (val/test identical to 100%), so
# --dataset-percent stays 100.
# detection3d.py runs the test F4 threshold sweep itself and writes results.json.
# =========================

set -u
: "${METHOD:?set METHOD}" "${DATASET:?set DATASET}" "${PERCENT:?set PERCENT}"

date
hostname
nvidia-smi

source ~/.bashrc
conda activate cryoet

cd /cluster/home/t129616uhn/projects/CryoDINO/3DINO || exit 1

# =========================
# Paths
# =========================
DETECTION_ROOT="/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets/downstream_detection"
EXPERIMENTS="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments"
BASE_OUTPUT_DIR="${EXPERIMENTS}/finetuning_detection"
CACHE_DIR_BASE="${EXPERIMENTS}/cache_dir_downstream_detection"
B200_WEIGHTS="${EXPERIMENTS}/ssl3d_run_b200_high_res_128/eval/training_6249/teacher_checkpoint.pth"

case "$DATASET" in
  czi) PATCH_DIR="${DETECTION_ROOT}/Dataset440_CZII_10440_detection_patches_128" ;;
  byu) PATCH_DIR="${DETECTION_ROOT}/byu_detection_patches_128" ;;
  *) echo "Unknown DATASET: $DATASET"; exit 1 ;;
esac

case "$PERCENT" in
  100)   BASE_DATA_DIR="$PATCH_DIR" ;;
  50|10) BASE_DATA_DIR="${PATCH_DIR}/${DATASET}_${PERCENT}percent" ;;
  *) echo "Unknown PERCENT: $PERCENT"; exit 1 ;;
esac

# loaders.py make_detection_dataset_3d reads "{base-data-dir}/{dataset_name}_100_datalist.json"
DATALIST="${BASE_DATA_DIR}/${DATASET}_100_datalist.json"
[ -f "$DATALIST" ] || { echo "Missing datalist: $DATALIST"; exit 1; }

EXTRA_ARGS=()
case "$METHOD" in
  cryodino)
    SEGMENTATION_HEAD="ViTAdapterUNETR"
    PRETRAINED_WEIGHTS="$B200_WEIGHTS"
    RUN_NAME="cryodino_b200_highres128_6249_${DATASET}_${PERCENT}pct_vit_adapter"
    ;;
  random_init)
    SEGMENTATION_HEAD="ViTAdapterUNETR"
    PRETRAINED_WEIGHTS=""   # empty = random initialization
    EXTRA_ARGS+=(--train-feature-model)
    RUN_NAME="random_init_${DATASET}_${PERCENT}pct_vit_adapter"
    ;;
  resnetfpn)
    SEGMENTATION_HEAD="ResNetFPN"
    PRETRAINED_WEIGHTS=""   # unused: no DINO backbone is built for this head
    RUN_NAME="resnetfpn_scratch_${DATASET}_${PERCENT}pct"
    ;;
  *) echo "Unknown METHOD: $METHOD"; exit 1 ;;
esac

# =========================
# Fixed Parameters (same budget for every method: 100 x 300 = 30k iters)
# =========================
CONFIG_FILE="dinov2/configs/train/vit3d_highres.yaml"
EPOCHS=100
EPOCH_LENGTH=300
EVAL_ITERS=1200
WARMUP_ITERS=3000
IMAGE_SIZE=128                # must match the 128^3 patch size for detection
BATCH_SIZE=4
NUM_WORKERS=24               # of 32 CPUs: loader workers (OMP=1 each); rest for the main process
LEARNING_RATE=1e-4
RESIZE_SCALE=1.0

OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"
CACHE_DIR="${CACHE_DIR_BASE}/${RUN_NAME}"

rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

echo "=============================================="
echo "Detection: method=$METHOD dataset=$DATASET percent=$PERCENT"
echo "Head: $SEGMENTATION_HEAD  extra: ${EXTRA_ARGS[*]:-none}"
echo "Pretrained weights: ${PRETRAINED_WEIGHTS:-<none>}"
echo "Datalist: $DATALIST"
echo "Output: $OUTPUT_DIR"
echo "Git: $(git -C .. rev-parse --short HEAD) on $(git -C .. rev-parse --abbrev-ref HEAD)"
echo "=============================================="

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python dinov2/eval/detection3d.py \
  --config-file "$CONFIG_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --pretrained-weights "$PRETRAINED_WEIGHTS" \
  --dataset-name "$DATASET" \
  --dataset-percent 100 \
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
  --resize-scale "$RESIZE_SCALE" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}   # empty-array safe under set -u on bash < 4.4

echo "Finished: $OUTPUT_DIR (results.json)"
date
