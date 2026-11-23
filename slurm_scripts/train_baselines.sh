#!/bin/bash
#SBATCH -J cryodino-baseline
#SBATCH -p gpu_pmccaiteam
#SBATCH -t 2-00:00:00
#SBATCH --account=pmccaiteam_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --output=slurm_logs/%x_%j.log

# =========================
# Segmentation baselines: UNet or UNETR from random initialization (baselines/unet_unetr/train.py,
# 100 epochs x 300 iterations, batch 2, 112^3, lr 1e-4) on one dataset, then test-set inference.
#
#   sbatch --export=ALL,MODEL=<unet|unetr>,DATASET=<001|010|989|049>,PERCENT=<100|50|10> train_baselines.sh
#   (submit_segmentation.sh BASELINES=1 submits all of them)
#
# The test set is always the full 100% datalist; 50%/10% only subsample training.
# =========================
set -euo pipefail
: "${CRYOET_ROOT:?set CRYOET_ROOT to the CryoDINO data folder (holding datasets/ and experiments/), e.g. export CRYOET_ROOT=/path/to/cryoet}"
: "${MODEL:?set MODEL (unet | unetr)}"
: "${DATASET:?set DATASET (001 | 010 | 989 | 049)}"
: "${PERCENT:=100}"
case "$MODEL" in unet|unetr) ;; *) echo "Unknown MODEL: $MODEL"; exit 1 ;; esac

REPO="${REPO:-/cluster/home/$USER/scripts/cryoet/CryoDINO}"
EXP="${EXP:-${CRYOET_ROOT}/experiments}"
BASE_DATA_DIR="$EXP"
CACHE_DIR_BASE="$EXP/cache_dir_downstream"
OUTPUT_BASE="$EXP/baselines"

case "$DATASET" in
  001) DS="Dataset001_CZII_10001_patches512";              N_CLASSES=4 ;;
  010) DS="Dataset010_CZII_10010_patches512";              N_CLASSES=2 ;;
  989) DS="Dataset989_EMPIAR_10989_transposed_patches512"; N_CLASSES=2 ;;
  049) DS="Dataset049_EMPIAR_12049_transposed_patches512"; N_CLASSES=4 ;;
  *) echo "Unknown DATASET: $DATASET"; exit 1 ;;
esac
case "$PERCENT" in 100) SUFFIX="" ;; 50|10) SUFFIX="_${PERCENT}percent" ;; *) echo "Unknown PERCENT: $PERCENT"; exit 1 ;; esac
MERGED=""; [ "$DATASET" = 049 ] && MERGED="_merged"   # see train_segmentation.sh

DATASET_NAME="${DS}${SUFFIX}"
CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DS}${MERGED}"
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}_${DATASET_NAME}"
IMAGE_SIZE=112
BATCH_SIZE=2

date
hostname
nvidia-smi || true
set +u; source ~/.bashrc; conda activate cryodino; set -u
echo "MODEL=$MODEL DATASET=$DATASET_NAME output=$OUTPUT_DIR"

cd "$REPO/3DINO" || exit 1
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python ../baselines/unet_unetr/train.py \
    --model_name "$MODEL" \
    --dataset_name "$DATASET_NAME" \
    --base_data_dir "$BASE_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --image_size "$IMAGE_SIZE" \
    --epochs 100 \
    --epoch_length 300 \
    --eval_iters 600 \
    --warmup_iters 3000 \
    --learning_rate 1e-4 \
    --batch_size "$BATCH_SIZE" \
    --num_workers 10 \
    --dataset_percent 100

CHECKPOINT="${OUTPUT_DIR}/best_model.pth"
INFER_OUTPUT_DIR="${OUTPUT_DIR}/inference"
if [ ! -f "$CHECKPOINT" ]; then
  echo "[SKIP inference] checkpoint not found: $CHECKPOINT"
  exit 1
fi
mkdir -p "$INFER_OUTPUT_DIR"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python ../baselines/unet_unetr/inference.py \
  --model-name "$MODEL" \
  --checkpoint "$CHECKPOINT" \
  --image-size "$IMAGE_SIZE" \
  --num-classes "$N_CLASSES" \
  --datalist "${BASE_DATA_DIR}/${DS}_100_datalist.json" \
  --dataset-name "$DS" \
  --output-dir "$INFER_OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --cpu-metrics
date
