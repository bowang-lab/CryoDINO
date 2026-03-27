#!/bin/bash
#SBATCH -J 3dino-ft-patches-ds-specific-aug-full-ft
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
# Full fine-tuning (unfrozen encoder + decoder): h100_high_res training_9374
# Pre-extracted 512^3 patches (.pt), N_crops-1 per patch (OOM fix), batch size 2
# ViTAdapterUNETR head, dataset-specific augmentations (Experiment 4), deep supervision
# =========================

date
hostname
pwd
nvidia-smi

source ~/.bashrc
conda activate cryoet

cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1

# =========================
# Fixed Parameters
# =========================
CONFIG_FILE="dinov2/configs/train/vit3d_highres.yaml"
PRETRAINED_WEIGHTS="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100_high_res/eval/training_9374/teacher_checkpoint.pth"
BASE_OUTPUT_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/finetuning"
DATASET_PERCENT=100
BASE_DATA_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments"
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

# Inference parameters — dataset-specific
declare -A DATALIST
DATALIST["Dataset001_CZII_10001_patches512"]="${BASE_DATA_DIR}/Dataset001_CZII_10001_patches512_100_datalist.json"
DATALIST["Dataset010_CZII_10010_patches512"]="${BASE_DATA_DIR}/Dataset010_CZII_10010_patches512_100_datalist.json"
DATALIST["Dataset989_EMPIAR_10989_transposed_patches512"]="${BASE_DATA_DIR}/Dataset989_EMPIAR_10989_transposed_patches512_100_datalist.json"
DATALIST["Dataset049_EMPIAR_12049_transposed_patches512"]="${BASE_DATA_DIR}/Dataset049_EMPIAR_12049_transposed_patches512_100_datalist.json"

declare -A NUM_CLASSES
NUM_CLASSES["Dataset001_CZII_10001_patches512"]=4
NUM_CLASSES["Dataset010_CZII_10010_patches512"]=2
NUM_CLASSES["Dataset989_EMPIAR_10989_transposed_patches512"]=2
NUM_CLASSES["Dataset049_EMPIAR_12049_transposed_patches512"]=4

declare -A INFER_DATASET_NAME
INFER_DATASET_NAME["Dataset001_CZII_10001_patches512"]="Dataset001_CZII_10001"
INFER_DATASET_NAME["Dataset010_CZII_10010_patches512"]="Dataset010_CZII_10010"
INFER_DATASET_NAME["Dataset989_EMPIAR_10989_transposed_patches512"]="Dataset989_EMPIAR_10989_transposed"
INFER_DATASET_NAME["Dataset049_EMPIAR_12049_transposed_patches512"]="Dataset049_EMPIAR_12049_transposed"

mkdir -p "$BASE_OUTPUT_DIR"

DATASETS=(
    "Dataset001_CZII_10001_patches512"
    "Dataset010_CZII_10010_patches512"
    "Dataset989_EMPIAR_10989_transposed_patches512"
    "Dataset049_EMPIAR_12049_transposed_patches512"
    )

for DATASET_NAME in "${DATASETS[@]}"; do

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}_vit_adapter_ds_specific_aug_full_ft"
    if [[ "$DATASET_NAME" == *"12049"* ]]; then
        CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}_merged"
    else
        CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}"
    fi

    mkdir -p "$CACHE_DIR"
    mkdir -p "$OUTPUT_DIR"

    echo "=============================================="
    echo "Starting 3D segmentation FULL fine-tuning (patches mode)..."
    echo "Config: $CONFIG_FILE"
    echo "Pretrained weights: $PRETRAINED_WEIGHTS"
    echo "Dataset: $DATASET_NAME"
    echo "Output directory: $OUTPUT_DIR"
    echo "=============================================="

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python dinov2/eval/segmentation3d.py \
      --config-file "$CONFIG_FILE" \
      --output-dir "$OUTPUT_DIR" \
      --pretrained-weights "$PRETRAINED_WEIGHTS" \
      --dataset-name "$DATASET_NAME" \
      --dataset-percent "$DATASET_PERCENT" \
      --base-data-dir "$BASE_DATA_DIR" \
      --segmentation-head "$SEGMENTATION_HEAD" \
      --train-feature-model \
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
    # Inference on test set immediately after training
    # =========================
    CHECKPOINT="${OUTPUT_DIR}/best_model.pth"
    INFER_OUTPUT_DIR="${OUTPUT_DIR}/inference"
    mkdir -p "$INFER_OUTPUT_DIR"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "  [SKIP inference] checkpoint not found: $CHECKPOINT"
    else
        echo "----------------------------------------------"
        echo "Running inference for: $DATASET_NAME"
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
          --num-classes "${NUM_CLASSES[$DATASET_NAME]}" \
          --datalist "${DATALIST[$DATASET_NAME]}" \
          --output-dir "$INFER_OUTPUT_DIR" \
          --dataset-name "${INFER_DATASET_NAME[$DATASET_NAME]}" \
          --overlap "$OVERLAP" \
          --batch-size "$BATCH_SIZE" \
          --cpu-metrics \
          --deep-supervision

        echo "Finished inference: $INFER_OUTPUT_DIR"

        cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1
    fi

done

echo "All datasets completed!"
date
