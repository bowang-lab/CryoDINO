#!/bin/bash
#SBATCH -J cryodino_3dino-ft-b200-best-ckpt-50pct
#SBATCH -p gpu_pmccaiteam
#SBATCH -t 4-00:00:00
#SBATCH --account=pmccaiteam_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

# =========================
# Fine-tuning with ViTAdapterUNETR head, 50% training data, all 4 datasets.
# Each dataset uses whichever B200 backbone won the 100% segmentation
# benchmark for it:
#   Dataset001/CZII-10001        -> highres112 / training_6249 (0.8918 vs 0.8912)
#   Dataset010/CZII-10010        -> highres128 / training_6249 (0.8474 vs 0.8289)
#   EMPIAR-10989                 -> highres128 / training_6249 (0.6758 vs 0.6555)
#   Dataset049/EMPIAR-12049      -> highres112 / training_6249 (0.4889 vs 0.4867)
# Pre-built "<base>_50percent_100_datalist.json" files carry the training
# subset (val/test = full 100% set), matching the h100 10pct/50pct convention
# — see train_3dino_ft_h100_highres_9374_patches_50pct.sh.
# =========================

date
hostname
pwd
nvidia-smi

source ~/.bashrc
conda activate cryodino

cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1

BASE_DATA_DIR="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments"
BASE_OUTPUT_DIR="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/finetuning"
CACHE_DIR_BASE="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/cache_dir_downstream"
DATASET_PERCENT=100
SEGMENTATION_HEAD="ViTAdapterUNETR"
EPOCHS=100
EPOCH_LENGTH=300
EVAL_ITERS=600
WARMUP_ITERS=3000
BATCH_SIZE=2
LEARNING_RATE=1e-4
RESIZE_SCALE=1.0
OVERLAP=0.75

HIGHRES112_CKPT="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/ssl3d_run_b200_high_res/eval/training_6249/teacher_checkpoint.pth"
HIGHRES128_CKPT="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/ssl3d_run_b200_high_res_128/eval/training_6249/teacher_checkpoint.pth"

# Best-checkpoint config per base dataset name
declare -A CONFIG_FILE
CONFIG_FILE["Dataset001_CZII_10001_patches512"]="dinov2/configs/train/vit3d_highres_112.yaml"
CONFIG_FILE["Dataset010_CZII_10010_patches512"]="dinov2/configs/train/vit3d_highres.yaml"
CONFIG_FILE["Dataset989_EMPIAR_10989_transposed_patches512"]="dinov2/configs/train/vit3d_highres.yaml"
CONFIG_FILE["Dataset049_EMPIAR_12049_transposed_patches512"]="dinov2/configs/train/vit3d_highres_112.yaml"

declare -A PRETRAINED_WEIGHTS
PRETRAINED_WEIGHTS["Dataset001_CZII_10001_patches512"]="$HIGHRES112_CKPT"
PRETRAINED_WEIGHTS["Dataset010_CZII_10010_patches512"]="$HIGHRES128_CKPT"
PRETRAINED_WEIGHTS["Dataset989_EMPIAR_10989_transposed_patches512"]="$HIGHRES128_CKPT"
PRETRAINED_WEIGHTS["Dataset049_EMPIAR_12049_transposed_patches512"]="$HIGHRES112_CKPT"

declare -A BACKBONE_TAG
BACKBONE_TAG["Dataset001_CZII_10001_patches512"]="ssl3d_run_b200_high_res"
BACKBONE_TAG["Dataset010_CZII_10010_patches512"]="ssl3d_run_b200_high_res_128"
BACKBONE_TAG["Dataset989_EMPIAR_10989_transposed_patches512"]="ssl3d_run_b200_high_res_128"
BACKBONE_TAG["Dataset049_EMPIAR_12049_transposed_patches512"]="ssl3d_run_b200_high_res"

declare -A IMAGE_SIZE
IMAGE_SIZE["Dataset001_CZII_10001_patches512"]=112
IMAGE_SIZE["Dataset010_CZII_10010_patches512"]=128
IMAGE_SIZE["Dataset989_EMPIAR_10989_transposed_patches512"]=128
IMAGE_SIZE["Dataset049_EMPIAR_12049_transposed_patches512"]=112

declare -A NUM_WORKERS
NUM_WORKERS["Dataset001_CZII_10001_patches512"]=4
NUM_WORKERS["Dataset010_CZII_10010_patches512"]=16
NUM_WORKERS["Dataset989_EMPIAR_10989_transposed_patches512"]=16
NUM_WORKERS["Dataset049_EMPIAR_12049_transposed_patches512"]=16

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

CACHE_DATASETS=(
    "Dataset001_CZII_10001_patches512"
    "Dataset010_CZII_10010_patches512"
    "Dataset989_EMPIAR_10989_transposed_patches512"
    "Dataset049_EMPIAR_12049_transposed_patches512"
    )

for CACHE_DATASET in "${CACHE_DATASETS[@]}"; do

    DATASET_NAME="${CACHE_DATASET}_50percent"

    CFG="${CONFIG_FILE[$CACHE_DATASET]}"
    CKPT="${PRETRAINED_WEIGHTS[$CACHE_DATASET]}"
    IMG_SIZE="${IMAGE_SIZE[$CACHE_DATASET]}"
    WORKERS="${NUM_WORKERS[$CACHE_DATASET]}"
    N_CLASSES="${NUM_CLASSES[$CACHE_DATASET]}"

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${BACKBONE_TAG[$CACHE_DATASET]}_training_6249_${DATASET_NAME}_vit_adapter"
    # Reuse the LP sweep's warm per-dataset cache (shared with the 100% run).
    CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${CACHE_DATASET}"
    if [[ "$CACHE_DATASET" == *"12049"* ]]; then
        CACHE_DIR="${CACHE_DIR}_merged"
    fi

    mkdir -p "$CACHE_DIR"
    mkdir -p "$OUTPUT_DIR"

    echo "=============================================="
    echo "Starting 3D segmentation fine-tuning (50% data)..."
    echo "Config: $CFG"
    echo "Pretrained weights: $CKPT"
    echo "Dataset: $DATASET_NAME"
    echo "Output directory: $OUTPUT_DIR"
    echo "Cache directory: $CACHE_DIR (shared with 100% run)"
    echo "=============================================="

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python dinov2/eval/segmentation3d.py \
      --config-file "$CFG" \
      --output-dir "$OUTPUT_DIR" \
      --pretrained-weights "$CKPT" \
      --dataset-name "$DATASET_NAME" \
      --dataset-percent "$DATASET_PERCENT" \
      --base-data-dir "$BASE_DATA_DIR" \
      --segmentation-head "$SEGMENTATION_HEAD" \
      --epochs "$EPOCHS" \
      --epoch-length "$EPOCH_LENGTH" \
      --eval-iters "$EVAL_ITERS" \
      --warmup-iters "$WARMUP_ITERS" \
      --image-size "$IMG_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$WORKERS" \
      --learning-rate "$LEARNING_RATE" \
      --cache-dir "$CACHE_DIR" \
      --resize-scale "$RESIZE_SCALE"

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
          --config-file "3DINO/${CFG}" \
          --pretrained-weights "$CKPT" \
          --checkpoint "$CHECKPOINT" \
          --segmentation-head "$SEGMENTATION_HEAD" \
          --image-size "$IMG_SIZE" \
          --num-classes "$N_CLASSES" \
          --datalist "${BASE_DATA_DIR}/${CACHE_DATASET}_100_datalist.json" \
          --output-dir "$INFER_OUTPUT_DIR" \
          --dataset-name "${INFER_DATASET_NAME[$CACHE_DATASET]}" \
          --overlap "$OVERLAP" \
          --batch-size "$BATCH_SIZE" \
          --cpu-metrics

        echo "Finished inference: $INFER_OUTPUT_DIR"

        cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1
    fi

done

echo "All datasets completed!"
date
