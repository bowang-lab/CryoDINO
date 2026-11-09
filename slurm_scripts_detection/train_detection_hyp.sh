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
#   cryodino_ft same as cryodino but the backbone is fine-tuned too (--train-feature-model)
#   cryodino_last1 same as cryodino but the last ViT block is unfrozen (--train-last-n-blocks 1)
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

# cryoet has torch 2.1 (cu121), which has no sm_100 kernels. On B200 nodes a venv layered on
# cryoet (--system-site-packages) supplies torch 2.7.1 cu128 + xformers 0.0.31; every other
# package still comes from cryoet. A100/H100 use cryoet as is.
B200_VENV="/cluster/projects/pmccaiteam/sumin/envs/cryoet-b200"

set +u   # /etc/bashrc and conda's activate scripts read unset variables
source ~/.bashrc
conda activate cryoet
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q B200; then
  source "$B200_VENV/bin/activate"
fi
set -u
echo "Python env: $(python -c 'import sys, torch; print(sys.prefix, "torch", torch.__version__)')"

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
  byu)
    # BYU_SPLIT=motoronly (default): split lists without the 286 motor-free tomograms, training
    # rebalanced 1:1 fg:bg. BYU_SPLIT=all: the original lists (same as train_detection.sh).
    BYU_SPLIT="${BYU_SPLIT:-motoronly}"
    case "$BYU_SPLIT" in
      motoronly) PATCH_DIR="${DETECTION_ROOT}/byu_detection_patches_128_motoronly" ;;
      all)       PATCH_DIR="${DETECTION_ROOT}/byu_detection_patches_128" ;;
      *) echo "Unknown BYU_SPLIT: $BYU_SPLIT (motoronly|all)"; exit 1 ;;
    esac ;;
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
  cryodino_ft)
    SEGMENTATION_HEAD="ViTAdapterUNETR"
    PRETRAINED_WEIGHTS="$B200_WEIGHTS"
    EXTRA_ARGS+=(--train-feature-model)
    RUN_NAME="cryodino_ft_b200_highres128_6249_${DATASET}_${PERCENT}pct_vit_adapter"
    ;;
  cryodino_last1)
    SEGMENTATION_HEAD="ViTAdapterUNETR"
    PRETRAINED_WEIGHTS="$B200_WEIGHTS"
    EXTRA_ARGS+=(--train-last-n-blocks 1)
    RUN_NAME="cryodino_last1_b200_highres128_6249_${DATASET}_${PERCENT}pct_vit_adapter"
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
EPOCHS="${EPOCHS:-100}"
EPOCH_LENGTH="${EPOCH_LENGTH:-300}"
EVAL_ITERS=1200
WARMUP_ITERS="${WARMUP_ITERS:-3000}"
IMAGE_SIZE=128                # must match the 128^3 patch size for detection
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-24}"   # of 32 CPUs: loader workers (OMP=1 each); rest for the main process.
                                  # Override with --export=...,NUM_WORKERS=8 when submitting with less --mem/--cpus-per-task
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
RESIZE_SCALE=1.0

# ViT decoder (ViTAdapterUNETR methods only), overridable via --export=...,DECODER_FUSION=fpn:
#   adapter (default) original ViT-Adapter + UNETR decoder; run name unchanged
#   fpn               plain ViT, 4 blocks (ENCODER_LAYERS, default 5,11,17,23) -> strides 2/4/8/16 -> FPN
#   concat            plain ViT, blocks (ENCODER_LAYERS, default -2,-1) concatenated -> simple pyramid -> FPN
#   adapter_multi     original adapter head + extra blocks (ENCODER_LAYERS, default 4,10,16,22) fused per
#                     adapter stage through zero-init 1x1 convs (starts identical to adapter)
#   adapter_fpn       original adapter head + zero-init FPN top-down residual over the UNETR skips
# Non-default settings add e.g. _fpnL5_11_17_23 / _concatL-2_-1 / _adapter_multiL4_10_16_22 /
# _adapter_fpn to the run name.
DECODER_FUSION="${DECODER_FUSION:-adapter}"
ENCODER_LAYERS="${ENCODER_LAYERS:-}"
ENCODER_LAYERS="${ENCODER_LAYERS// /}"
case "$DECODER_FUSION" in
  adapter)
    [ -z "$ENCODER_LAYERS" ] || { echo "ENCODER_LAYERS needs DECODER_FUSION=fpn|concat"; exit 1; } ;;
  adapter_fpn)
    [ "$SEGMENTATION_HEAD" = "ViTAdapterUNETR" ] || { echo "DECODER_FUSION=$DECODER_FUSION needs a ViT method, not $METHOD"; exit 1; }
    [ -z "$ENCODER_LAYERS" ] || { echo "DECODER_FUSION=adapter_fpn takes no ENCODER_LAYERS"; exit 1; }
    EXTRA_ARGS+=(--decoder-fusion "$DECODER_FUSION")
    RUN_NAME="${RUN_NAME}_${DECODER_FUSION}"
    ;;
  fpn|concat|adapter_multi)
    [ "$SEGMENTATION_HEAD" = "ViTAdapterUNETR" ] || { echo "DECODER_FUSION=$DECODER_FUSION needs a ViT method, not $METHOD"; exit 1; }
    if [ -z "$ENCODER_LAYERS" ]; then
      case "$DECODER_FUSION" in
        fpn) ENCODER_LAYERS="5,11,17,23" ;;
        concat) ENCODER_LAYERS="-2,-1" ;;
        adapter_multi) ENCODER_LAYERS="4,10,16,22" ;;
      esac
    fi
    [[ "$ENCODER_LAYERS" =~ ^-?[0-9]+(,-?[0-9]+)*$ ]] || { echo "Bad ENCODER_LAYERS: $ENCODER_LAYERS"; exit 1; }
    EXTRA_ARGS+=(--decoder-fusion "$DECODER_FUSION" "--encoder-layers=$ENCODER_LAYERS")   # = form: argparse rejects a separate "-2,-1"
    RUN_NAME="${RUN_NAME}_${DECODER_FUSION}L${ENCODER_LAYERS//,/_}"
    ;;
  *) echo "Unknown DECODER_FUSION: $DECODER_FUSION"; exit 1 ;;
esac

# Hyperparameters (overridable via --export=...,LEARNING_RATE=1e-5 etc.) go into the run name so
# runs never share an output/cache dir; refuse to start over an existing run.
[ "$DATASET" = byu ] && [ "$BYU_SPLIT" = motoronly ] && RUN_NAME="${RUN_NAME}_motoronly"
RUN_NAME="${RUN_NAME}_lr${LEARNING_RATE}_bs${BATCH_SIZE}_ep${EPOCHS}x${EPOCH_LENGTH}_wu${WARMUP_ITERS}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"
if [ -e "$OUTPUT_DIR" ]; then echo "Output dir exists, not overwriting: $OUTPUT_DIR"; exit 1; fi
CACHE_DIR="${CACHE_DIR_BASE}/${RUN_NAME}"

rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

echo "=============================================="
echo "Detection: method=$METHOD dataset=$DATASET percent=$PERCENT"
echo "Head: $SEGMENTATION_HEAD  extra: ${EXTRA_ARGS[*]:-none}"
echo "Pretrained weights: ${PRETRAINED_WEIGHTS:-<none>}"
echo "Datalist: $DATALIST"
[ "$DATASET" = byu ] && echo "BYU split: $BYU_SPLIT"
echo "Output: $OUTPUT_DIR"
echo "Hyperparams: lr=$LEARNING_RATE batch=$BATCH_SIZE epochs=$EPOCHS x $EPOCH_LENGTH warmup=$WARMUP_ITERS"
[ "$SEGMENTATION_HEAD" = "ViTAdapterUNETR" ] && echo "ViT decoder: fusion=$DECODER_FUSION encoder_layers=${ENCODER_LAYERS:-<adapter stages, blocks 5,11,17,23>}"
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
STATUS=$?

if [ "$STATUS" -eq 0 ] && [ -f "$OUTPUT_DIR/results.json" ]; then
  echo "Finished: $OUTPUT_DIR/results.json"
else
  echo "FAILED (exit $STATUS): no results.json in $OUTPUT_DIR"
  [ "$STATUS" -eq 0 ] && STATUS=1
fi
date
exit "$STATUS"
