#!/bin/bash
#SBATCH -J cryodino_3dino-ft-b200-6249-j3
#SBATCH -p gpu_pmcc_ai_team
#SBATCH -t 2-00:00:00
#SBATCH --account=pmcc_ai_team_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

# =========================
# Dataset989_EMPIAR_10989_transposed — fine-tuning with ViTAdapterUNETR head.
# Backbone: B200 ssl3d_run_b200_high_res / training_6249 — the best overall LP
# checkpoint from the linear_probing_h100_b200_comparison sweep. NOTE: this
# dataset's LP scores collapsed to near-zero foreground Dice across almost
# every backbone (including random init) — worth watching whether ViTAdapter-
# UNETR (a stronger head) breaks that collapse or not.
# Plain (non mix-patch-augmented) dataset, matching the LP sweep's inputs
# exactly for a direct comparison — no mix_patches_augmentation_cryodino.py.
# =========================

date
hostname
pwd
nvidia-smi

source ~/.bashrc
conda activate cryodino

cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1

BASE_DATA_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments"
DATASET_NAME="Dataset989_EMPIAR_10989_transposed_patches512"
NUM_CLASSES=2
INFER_DS_NAME="Dataset989_EMPIAR_10989_transposed"

# =========================
# Fixed training parameters
# =========================
CONFIG_FILE="dinov2/configs/train/vit3d_highres_112.yaml"
PRETRAINED_WEIGHTS="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_b200_high_res/eval/training_6249/teacher_checkpoint.pth"
BASE_OUTPUT_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/finetuning"
DATASET_PERCENT=100
SEGMENTATION_HEAD="ViTAdapterUNETR"
EPOCHS=100
EPOCH_LENGTH=300
EVAL_ITERS=600
WARMUP_ITERS=3000
IMAGE_SIZE=112
BATCH_SIZE=2
NUM_WORKERS=16
LEARNING_RATE=1e-4
CACHE_DIR_BASE="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/cache_dir_downstream"
RESIZE_SCALE=1.0
OVERLAP=0.75

OUTPUT_DIR="${BASE_OUTPUT_DIR}/ssl3d_run_b200_high_res_training_6249_${DATASET_NAME}_vit_adapter"
CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}"

mkdir -p "$CACHE_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Fine-tuning: $DATASET_NAME"
echo "Pretrained weights: $PRETRAINED_WEIGHTS"
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
      --datalist "${BASE_DATA_DIR}/${DATASET_NAME}_100_datalist.json" \
      --output-dir "$INFER_OUTPUT_DIR" \
      --dataset-name "$INFER_DS_NAME" \
      --overlap "$OVERLAP" \
      --batch-size "$BATCH_SIZE" \
      --cpu-metrics

    echo "Finished inference: $INFER_OUTPUT_DIR"

    cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1
fi

echo "Done!"
date
