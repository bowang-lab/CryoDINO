#!/bin/bash
#SBATCH -J 3dino-ft-patches-ds-specific-aug-mix-j1
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
# Dataset001_CZII_10001 — mix-patch augmentation + fine-tuning
# =========================

date
hostname
pwd
nvidia-smi

source ~/.bashrc
conda activate cryoet

cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO || exit 1

BASE_DATA_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments"
DATASETS_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream"

AUG_DS="Dataset001_CZII_10001_patches512"
AUG_INPUT="${BASE_DATA_DIR}/Dataset001_CZII_10001_patches512_100_datalist.json"
AUG_OUTDIR="${DATASETS_DIR}/Dataset001_CZII_10001_train_patches_512_augmented"
AUG_CLS=4
DATASET_NAME="Dataset001_CZII_10001_patches512_augmented"
NUM_CLASSES=4
INFER_DS_NAME="Dataset001_CZII_10001"

# =========================
# Augment patches
# =========================
echo "=============================================="
echo "Augmenting: $AUG_DS"
echo "=============================================="

# rm -rf "$AUG_OUTDIR"
mkdir -p "$AUG_OUTDIR"

AUG_JSON_NAME="$(basename "$AUG_INPUT" .json)_augmented.json"

python preprocessing/mix_patches_augmentation_cryodino.py \
    --datalist "$AUG_INPUT" \
    --output-dir "$AUG_OUTDIR" \
    --num-classes "$AUG_CLS" \
    --target-multiplier 3.0 \
    --max-patches 1000 \
    --seed 42

mv "${AUG_OUTDIR}/${AUG_JSON_NAME}" "${BASE_DATA_DIR}/${AUG_DS}_augmented_100_datalist.json"
echo "Moved augmented datalist → ${BASE_DATA_DIR}/${AUG_DS}_augmented_100_datalist.json"

cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1

# =========================
# Fixed training parameters
# =========================
CONFIG_FILE="dinov2/configs/train/vit3d_highres.yaml"
PRETRAINED_WEIGHTS="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100_high_res/eval/training_9374/teacher_checkpoint.pth"
BASE_OUTPUT_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/finetuning"
DATASET_PERCENT=100
SEGMENTATION_HEAD="ViTAdapterUNETR"
EPOCHS=100
EPOCH_LENGTH=300
EVAL_ITERS=600
WARMUP_ITERS=3000
IMAGE_SIZE=112
BATCH_SIZE=2
NUM_WORKERS=10
LEARNING_RATE=1e-4
CACHE_DIR_BASE="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/cache_dir_downstream"
RESIZE_SCALE=1.0
OVERLAP=0.75

OUTPUT_DIR="${BASE_OUTPUT_DIR}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}_vit_adapter_ds_specific_aug_mix_patches"
CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}"

# rm -rf "$CACHE_DIR"
# rm -rf "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Fine-tuning: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"
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
# Inference
# =========================
CHECKPOINT="${OUTPUT_DIR}/best_model.pth"
INFER_OUTPUT_DIR="${OUTPUT_DIR}/inference"
mkdir -p "$INFER_OUTPUT_DIR"

if [ ! -f "$CHECKPOINT" ]; then
    echo "  [SKIP inference] checkpoint not found: $CHECKPOINT"
else
    cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO || exit 1

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python inference/segmentation3d_inference.py \
      --config-file "3DINO/${CONFIG_FILE}" \
      --pretrained-weights "$PRETRAINED_WEIGHTS" \
      --checkpoint "$CHECKPOINT" \
      --segmentation-head "$SEGMENTATION_HEAD" \
      --image-size "$IMAGE_SIZE" \
      --num-classes "$NUM_CLASSES" \
      --datalist "${BASE_DATA_DIR}/${AUG_DS}_augmented_100_datalist.json" \
      --output-dir "$INFER_OUTPUT_DIR" \
      --dataset-name "$INFER_DS_NAME" \
      --overlap "$OVERLAP" \
      --batch-size "$BATCH_SIZE" \
      --cpu-metrics

    echo "Finished inference: $INFER_OUTPUT_DIR"
fi

echo "Done!"
date
