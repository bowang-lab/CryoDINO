#!/bin/bash
#SBATCH -J cryodino_3dino-ft-b200-6249-j4
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
# Dataset049_EMPIAR_12049_transposed — fine-tuning with ViTAdapterUNETR head.
# Backbone: B200 ssl3d_run_b200_high_res / training_6249 — the best overall LP
# checkpoint from the linear_probing_h100_b200_comparison sweep. NOTE: in the
# LP sweep this dataset actually favored highres128/6249 over highres112/6249
# by the largest relative margin (~11%) of any dataset — worth also trying
# ssl3d_run_b200_high_res_128/training_6249 (config vit3d_highres.yaml, 128)
# here if this run underperforms H100.
# Plain (non mix-patch-augmented) dataset — loaders.py/augmentations.py apply
# the 6-class -> 4-class background remap (_remap_12049) automatically for any
# dataset name containing "12049", matching the LP sweep's inputs exactly.
# =========================

date
hostname
pwd
nvidia-smi

source ~/.bashrc
conda activate cryodino

cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1

BASE_DATA_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments"
DATASET_NAME="Dataset049_EMPIAR_12049_transposed_patches512"
NUM_CLASSES=4
INFER_DS_NAME="Dataset049_EMPIAR_12049_transposed"

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
CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}_merged"

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
  --resize-scale "$RESIZE_SCALE" \
  --deep-supervision

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
      --cpu-metrics \
      --deep-supervision

    echo "Finished inference: $INFER_OUTPUT_DIR"

    cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1
fi

echo "Done!"
date
