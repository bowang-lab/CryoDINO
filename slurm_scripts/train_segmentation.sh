#!/bin/bash
#SBATCH -J cryodino-seg
#SBATCH -p gpu_pmccaiteam
#SBATCH -t 2-00:00:00
#SBATCH --account=pmccaiteam_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --output=slurm_logs/%x_%j.log

# =========================
# CryoDINO segmentation fine-tuning (segmentation3d.py) on one dataset, then test-set inference
# with the best checkpoint (inference/segmentation3d_inference.py). One job = one run.
#
#   sbatch --export=ALL,PRESET=<preset>,DATASET=<001|010|989|049>,PERCENT=<100|50|10> train_segmentation.sh
#   (submit_segmentation.sh loops over datasets/percentages and sets memory/time per run)
#
# PRESET (backbone + training mode; ViTAdapterUNETR head, 100 epochs x 300 iterations, batch 2):
#   h100_9374        frozen H100 high-res backbone (training_9374), cryoet env, 112^3.
#                    100%: trains on the mix-patch augmented datalist (AUGMENT=1 regenerates it).
#   b200_112         frozen B200 high-res 112 backbone (training_6249), 100% only
#   b200_128         frozen B200 high-res 128 backbone (training_6249), 128^3, 100% only
#   b200_best        frozen, per dataset the B200 backbone that won at 100%
#                    (001, 049 -> b200_112; 010, 989 -> b200_128), 50% / 10%
#   b200_best_full   as b200_best but the encoder is trained too (--train-feature-model, lr 1e-5), 100%
#   random_init      same ViT, randomly initialised and trained (--train-feature-model), 112^3
#
# Output directory names are the ones the reported runs used, so results can be found from here.
# The test set is always the full 100% datalist; 50%/10% only subsample training
# (<dataset>_<p>percent_100_datalist.json).
# =========================
set -euo pipefail
: "${PRESET:?set PRESET (h100_9374 | b200_112 | b200_128 | b200_best | b200_best_full | random_init)}"
: "${DATASET:?set DATASET (001 | 010 | 989 | 049)}"
: "${PERCENT:=100}"
: "${AUGMENT:=0}"   # h100_9374 at 100% only: regenerate the mix-patch augmented datalist first
: "${NUM_WORKERS:=}" # override the per-run default below
: "${DEEP_SUPERVISION:=0}"  # 1: add --deep-supervision (the all-dataset b200_112 loop used it; the per-dataset runs did not)

REPO="${REPO:-/cluster/home/$USER/scripts/cryoet/CryoDINO}"
EXP="${EXP:-/cluster/projects/pmccaiteam/<USERNAME>/projects/cryoet/experiments}"
BASE_DATA_DIR="$EXP"
BASE_OUTPUT_DIR="$EXP/finetuning"
CACHE_DIR_BASE="$EXP/cache_dir_downstream"

# ---- datasets ----
case "$DATASET" in
  001) DS="Dataset001_CZII_10001_patches512";               N_CLASSES=4; AUG_CLS=4; INFER_NAME="Dataset001_CZII_10001";              AUG_MULT=2.0;  AUG_EMPTY=0.15
       AUG_OUTDIR_NAME="Dataset001_CZII_10001_train_patches_512_augmented" ;;
  010) DS="Dataset010_CZII_10010_patches512";               N_CLASSES=2; AUG_CLS=2; INFER_NAME="Dataset010_CZII_10010";              AUG_MULT=3.0;  AUG_EMPTY=0.02
       AUG_OUTDIR_NAME="Dataset010_CZII_10010_train_patches_512_augmented" ;;
  989) DS="Dataset989_EMPIAR_10989_transposed_patches512";  N_CLASSES=2; AUG_CLS=2; INFER_NAME="Dataset989_EMPIAR_10989_transposed"; AUG_MULT=20.0; AUG_EMPTY=0.02
       AUG_OUTDIR_NAME="Dataset989_EMPIAR_10989_transposed_train_patches_512_augmented" ;;
  # 12049: raw patches have 6 classes (0-5); training remaps them to 4 (augmentations.py)
  049) DS="Dataset049_EMPIAR_12049_transposed_patches512";  N_CLASSES=4; AUG_CLS=6; INFER_NAME="Dataset049_EMPIAR_12049_transposed"; AUG_MULT=3.0;  AUG_EMPTY=0.02
       AUG_OUTDIR_NAME="Dataset049_EMPIAR_12049_transposed_train_patches_512_augmented" ;;
  *) echo "Unknown DATASET: $DATASET"; exit 1 ;;
esac
case "$PERCENT" in 100) SUFFIX="" ;; 50|10) SUFFIX="_${PERCENT}percent" ;; *) echo "Unknown PERCENT: $PERCENT"; exit 1 ;; esac
# The 12049 cache carries a "_merged" suffix: its label remap runs before caching, so a shared
# non-suffixed cache could hold stale 6-class labels.
MERGED=""; [ "$DATASET" = 049 ] && MERGED="_merged"

# ---- backbones ----
H100_9374="$EXP/ssl3d_run_h100_high_res/eval/training_9374/teacher_checkpoint.pth"
B200_112="$EXP/ssl3d_run_b200_high_res/eval/training_6249/teacher_checkpoint.pth"
B200_128="$EXP/ssl3d_run_b200_high_res_128/eval/training_6249/teacher_checkpoint.pth"
CFG_112="dinov2/configs/train/vit3d_highres_112.yaml"
CFG_HIGHRES="dinov2/configs/train/vit3d_highres.yaml"   # 112^3 for H100, 128^3 for the B200 128 backbone
use_b200_112() { CONFIG_FILE="$CFG_112";     PRETRAINED_WEIGHTS="$B200_112"; IMAGE_SIZE=112; TAG="ssl3d_run_b200_high_res"; }
use_b200_128() { CONFIG_FILE="$CFG_HIGHRES"; PRETRAINED_WEIGHTS="$B200_128"; IMAGE_SIZE=128; TAG="ssl3d_run_b200_high_res_128"; }
use_b200_best() { case "$DATASET" in 001|049) use_b200_112 ;; 010|989) use_b200_128 ;; esac; }

ENV_NAME=cryodino
LEARNING_RATE=1e-4
TRAIN_FEATURE_MODEL=0
ALLOC_CONF=""
WORKERS_DEFAULT=16; [ "$DATASET" = 001 ] && WORKERS_DEFAULT=4   # Dataset001 ran out of memory with more workers
DATASET_NAME="${DS}${SUFFIX}"
INFER_DATALIST="${BASE_DATA_DIR}/${DS}_100_datalist.json"
CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DS}${MERGED}"   # shared by every run

need_percent() { case " $* " in *" $PERCENT "*) ;; *) echo "PRESET=$PRESET supports PERCENT in: $*"; exit 1 ;; esac; }
case "$PRESET" in
  h100_9374)
    ENV_NAME=cryoet; CONFIG_FILE="$CFG_HIGHRES"; PRETRAINED_WEIGHTS="$H100_9374"; IMAGE_SIZE=112
    WORKERS_DEFAULT=10
    if [ "$PERCENT" = 100 ]; then
      DATASET_NAME="${DS}_augmented"
      INFER_DATALIST="${BASE_DATA_DIR}/${DS}_augmented_100_datalist.json"
      CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}${MERGED}"
      OUTPUT_DIR="${BASE_OUTPUT_DIR}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}_vit_adapter_ds_specific_aug_mix_patches_v2"
    else
      CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DS}_merged"
      OUTPUT_DIR="${BASE_OUTPUT_DIR}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}_vit_adapter_pretrain_aug_merged"
    fi ;;
  b200_112)
    need_percent 100; use_b200_112
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TAG}_training_6249_${DATASET_NAME}_vit_adapter" ;;
  b200_128)
    need_percent 100; use_b200_128
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TAG}_training_6249_${DATASET_NAME}_vit_adapter_newaug" ;;
  b200_best)
    need_percent 50 10; use_b200_best
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TAG}_training_6249_${DATASET_NAME}_vit_adapter" ;;
  b200_best_full)
    need_percent 100; use_b200_best
    TRAIN_FEATURE_MODEL=1; LEARNING_RATE=1e-5; WORKERS_DEFAULT=4; ALLOC_CONF="expandable_segments:True"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TAG}_training_6249_${DATASET_NAME}_vit_adapter_full_ft" ;;
  random_init)
    CONFIG_FILE="$CFG_112"; PRETRAINED_WEIGHTS=""; IMAGE_SIZE=112
    TRAIN_FEATURE_MODEL=1; WORKERS_DEFAULT=7; ALLOC_CONF="expandable_segments:True"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/random_init_${DATASET_NAME}_vit_adapter" ;;
  *) echo "Unknown PRESET: $PRESET"; exit 1 ;;
esac
NUM_WORKERS="${NUM_WORKERS:-$WORKERS_DEFAULT}"
SEGMENTATION_HEAD="ViTAdapterUNETR"
BATCH_SIZE=2
OVERLAP=0.75
FEATURE_ARG=(); [ "$TRAIN_FEATURE_MODEL" = 1 ] && FEATURE_ARG=(--train-feature-model)
INFER_ARG=()
[ "$DEEP_SUPERVISION" = 1 ] && { FEATURE_ARG+=(--deep-supervision); INFER_ARG=(--deep-supervision); }
[ -n "$ALLOC_CONF" ] && export PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF"

date
hostname
nvidia-smi || true
set +u; source ~/.bashrc; conda activate "$ENV_NAME"; set -u
echo "PRESET=$PRESET DATASET=$DATASET_NAME PERCENT=$PERCENT"
echo "weights=${PRETRAINED_WEIGHTS:-<random init>} config=$CONFIG_FILE image=$IMAGE_SIZE lr=$LEARNING_RATE workers=$NUM_WORKERS"
echo "output=$OUTPUT_DIR"
echo "cache=$CACHE_DIR"

# ---- optional: regenerate the mix-patch augmented datalist (h100_9374, 100%) ----
if [ "$AUGMENT" = 1 ]; then
  [ "$PRESET" = h100_9374 ] && [ "$PERCENT" = 100 ] || { echo "AUGMENT=1 only applies to PRESET=h100_9374 PERCENT=100"; exit 1; }
  cd "$REPO" || exit 1
  AUG_INPUT="${BASE_DATA_DIR}/${DS}_100_datalist.json"
  AUG_OUTDIR="${EXP%/experiments}/datasets/downstream/${AUG_OUTDIR_NAME}"
  rm -rf "$AUG_OUTDIR"; mkdir -p "$AUG_OUTDIR"
  python preprocessing/mix_patches_augmentation_cryodino.py \
      --datalist "$AUG_INPUT" \
      --output-dir "$AUG_OUTDIR" \
      --num-classes "$AUG_CLS" \
      --target-multiplier "$AUG_MULT" \
      --empty-threshold "$AUG_EMPTY" \
      --max-patches 1000 \
      --seed 42
  mv "${AUG_OUTDIR}/$(basename "$AUG_INPUT" .json)_augmented.json" "$INFER_DATALIST"
  rm -rf "$CACHE_DIR"   # the cache holds the previous augmented patches
fi

# ---- train ----
cd "$REPO/3DINO" || exit 1
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python dinov2/eval/segmentation3d.py \
  --config-file "$CONFIG_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --pretrained-weights "$PRETRAINED_WEIGHTS" \
  --dataset-name "$DATASET_NAME" \
  --dataset-percent 100 \
  --base-data-dir "$BASE_DATA_DIR" \
  --segmentation-head "$SEGMENTATION_HEAD" \
  ${FEATURE_ARG[@]+"${FEATURE_ARG[@]}"} \
  --epochs 100 \
  --epoch-length 300 \
  --eval-iters 600 \
  --warmup-iters 3000 \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --learning-rate "$LEARNING_RATE" \
  --cache-dir "$CACHE_DIR" \
  --resize-scale 1.0
echo "Finished training: $OUTPUT_DIR"

# ---- test-set inference with the best checkpoint ----
CHECKPOINT="${OUTPUT_DIR}/best_model.pth"
INFER_OUTPUT_DIR="${OUTPUT_DIR}/inference"
if [ ! -f "$CHECKPOINT" ]; then
  echo "[SKIP inference] checkpoint not found: $CHECKPOINT"
  exit 1
fi
mkdir -p "$INFER_OUTPUT_DIR"
cd "$REPO" || exit 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python inference/segmentation3d_inference.py \
  --config-file "3DINO/${CONFIG_FILE}" \
  --pretrained-weights "$PRETRAINED_WEIGHTS" \
  --checkpoint "$CHECKPOINT" \
  --segmentation-head "$SEGMENTATION_HEAD" \
  --image-size "$IMAGE_SIZE" \
  --num-classes "$N_CLASSES" \
  --datalist "$INFER_DATALIST" \
  --output-dir "$INFER_OUTPUT_DIR" \
  --dataset-name "$INFER_NAME" \
  --overlap "$OVERLAP" \
  --batch-size "$BATCH_SIZE" \
  ${INFER_ARG[@]+"${INFER_ARG[@]}"} \
  --cpu-metrics
echo "Finished inference: $INFER_OUTPUT_DIR"
date
