#!/bin/bash
#SBATCH -J 3dino-ft-patches-ds-specific-aug-mix
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
# Fine-tuning: h100_high_res training_9374
# Pre-extracted 512^3 patches (.pt), 4 crop per patch, batch size 2
# Frozen ViT, Dataset001 + Dataset010, EMPIAR_10989, EMPIAR_12049, with ViTAdapterUNETR head
# =========================

date
hostname
pwd
nvidia-smi

source ~/.bashrc
conda activate cryoet

cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO || exit 1

# =========================
# Re-generate patches (all patches, no fg threshold, with zscore)
# =========================
# PATCH_DATALISTS=(
#     # "/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/Dataset001_CZII_10001_100_datalist.json"
#     # "/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/Dataset010_CZII_10010_100_datalist.json"
#     # "/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/Dataset989_EMPIAR_10989_transposed_100_datalist.json"
#     "/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/Dataset049_EMPIAR_12049_transposed_100_datalist.json"
# )
# PATCH_OUTPUT_DIRS=(
#     # "/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream/Dataset001_CZII_10001_train_patches_512"
#     # "/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream/Dataset010_CZII_10010_train_patches_512"
#     # "/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream/Dataset989_EMPIAR_10989_transposed_train_patches_512"
#     "/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream/Dataset049_EMPIAR_12049_transposed_train_patches_512"
# )

# for i in "${!PATCH_DATALISTS[@]}"; do
#     echo "=============================================="
#     echo "Generating patches: ${PATCH_DATALISTS[$i]}"
#     echo "=============================================="

#     # Clean up old patches and cache
#     rm -rf "${PATCH_OUTPUT_DIRS[$i]}"

#     # Derive output JSON name with patches512 suffix
#     DATALIST_DIR=$(dirname "${PATCH_DATALISTS[$i]}")
#     DATALIST_BASE=$(basename "${PATCH_DATALISTS[$i]}" _100_datalist.json)
#     OUTPUT_JSON="${DATALIST_DIR}/${DATALIST_BASE}_patches512_100_datalist.json"

#     python preprocessing/downstream_patch_generation.py \
#         --zscore \
#         --fg-threshold 0.0 \
#         --patch-size 512 \
#         --datalist-json "${PATCH_DATALISTS[$i]}" \
#         --output-dir "${PATCH_OUTPUT_DIRS[$i]}" \
#         --output-json "$OUTPUT_JSON" \

# done

# =========================
# Augment patches for class balance (all datasets)
# =========================
BASE_DATA_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments"
DATASETS_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream"

declare -A AUG_INPUT_DATALISTS
AUG_INPUT_DATALISTS["Dataset001_CZII_10001_patches512"]="${BASE_DATA_DIR}/Dataset001_CZII_10001_patches512_100_datalist.json"
AUG_INPUT_DATALISTS["Dataset010_CZII_10010_patches512"]="${BASE_DATA_DIR}/Dataset010_CZII_10010_patches512_100_datalist.json"
AUG_INPUT_DATALISTS["Dataset989_EMPIAR_10989_transposed_patches512"]="${BASE_DATA_DIR}/Dataset989_EMPIAR_10989_transposed_patches512_100_datalist.json"
AUG_INPUT_DATALISTS["Dataset049_EMPIAR_12049_transposed_patches512"]="${BASE_DATA_DIR}/Dataset049_EMPIAR_12049_transposed_patches512_100_datalist.json"

declare -A AUG_PATCH_OUTDIRS
AUG_PATCH_OUTDIRS["Dataset001_CZII_10001_patches512"]="${DATASETS_DIR}/Dataset001_CZII_10001_train_patches_512_augmented"
AUG_PATCH_OUTDIRS["Dataset010_CZII_10010_patches512"]="${DATASETS_DIR}/Dataset010_CZII_10010_train_patches_512_augmented"
AUG_PATCH_OUTDIRS["Dataset989_EMPIAR_10989_transposed_patches512"]="${DATASETS_DIR}/Dataset989_EMPIAR_10989_transposed_train_patches_512_augmented"
AUG_PATCH_OUTDIRS["Dataset049_EMPIAR_12049_transposed_patches512"]="${DATASETS_DIR}/Dataset049_EMPIAR_12049_transposed_train_patches_512_augmented"

declare -A AUG_NUM_CLASSES
AUG_NUM_CLASSES["Dataset001_CZII_10001_patches512"]=4
AUG_NUM_CLASSES["Dataset010_CZII_10010_patches512"]=2
AUG_NUM_CLASSES["Dataset989_EMPIAR_10989_transposed_patches512"]=2
AUG_NUM_CLASSES["Dataset049_EMPIAR_12049_transposed_patches512"]=6

AUG_DATASETS=(
    "Dataset001_CZII_10001_patches512"
    "Dataset010_CZII_10010_patches512"
    "Dataset989_EMPIAR_10989_transposed_patches512"
    "Dataset049_EMPIAR_12049_transposed_patches512"
)

for AUG_DS in "${AUG_DATASETS[@]}"; do
    AUG_INPUT="${AUG_INPUT_DATALISTS[$AUG_DS]}"
    AUG_OUTDIR="${AUG_PATCH_OUTDIRS[$AUG_DS]}"
    AUG_CLS="${AUG_NUM_CLASSES[$AUG_DS]}"
    AUG_JSON_NAME="$(basename "$AUG_INPUT" .json)_augmented.json"

    echo "=============================================="
    echo "Augmenting patches: $AUG_DS"
    echo "=============================================="

    # rm -rf "$AUG_OUTDIR"
    mkdir -p "$AUG_OUTDIR"

    python preprocessing/mix_patches_augmentation_cryodino.py \
        --datalist "$AUG_INPUT" \
        --output-dir "$AUG_OUTDIR" \
        --num-classes "$AUG_CLS" \
        --target-multiplier 3.0 \
        --max-patches 1000 \
        --seed 42

    # Move augmented datalist JSON to BASE_DATA_DIR for consistency
    mv "${AUG_OUTDIR}/${AUG_JSON_NAME}" "${BASE_DATA_DIR}/${AUG_JSON_NAME}"
    echo "Moved augmented datalist → ${BASE_DATA_DIR}/${AUG_JSON_NAME}"
done

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
# Original (no mix-patch augmentation):
# DATALIST["Dataset001_CZII_10001_patches512"]="${BASE_DATA_DIR}/Dataset001_CZII_10001_patches512_100_datalist.json"
# DATALIST["Dataset010_CZII_10010_patches512"]="${BASE_DATA_DIR}/Dataset010_CZII_10010_patches512_100_datalist.json"
# DATALIST["Dataset989_EMPIAR_10989_transposed_patches512"]="${BASE_DATA_DIR}/Dataset989_EMPIAR_10989_transposed_patches512_100_datalist.json"
# DATALIST["Dataset049_EMPIAR_12049_transposed_patches512"]="${BASE_DATA_DIR}/Dataset049_EMPIAR_12049_transposed_patches512_100_datalist.json"
# Mix-patch augmented datalists (generated above, moved to BASE_DATA_DIR):
DATALIST["Dataset001_CZII_10001_patches512"]="${BASE_DATA_DIR}/Dataset001_CZII_10001_patches512_100_datalist_augmented.json"
DATALIST["Dataset010_CZII_10010_patches512"]="${BASE_DATA_DIR}/Dataset010_CZII_10010_patches512_100_datalist_augmented.json"
DATALIST["Dataset989_EMPIAR_10989_transposed_patches512"]="${BASE_DATA_DIR}/Dataset989_EMPIAR_10989_transposed_patches512_100_datalist_augmented.json"
DATALIST["Dataset049_EMPIAR_12049_transposed_patches512"]="${BASE_DATA_DIR}/Dataset049_EMPIAR_12049_transposed_patches512_100_datalist_augmented.json"

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

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}_vit_adapter_ds_specific_aug_mix_patches"
    if [[ "$DATASET_NAME" == *"12049"* ]]; then
        CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}_merged"
    else
        CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}"
    fi

    # Clean old cache and output
    # rm -rf "$CACHE_DIR"
    # rm -rf "$OUTPUT_DIR"
    mkdir -p "$CACHE_DIR"
    mkdir -p "$OUTPUT_DIR"

    echo "=============================================="
    echo "Starting 3D segmentation fine-tuning (patches mode)..."
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
          --cpu-metrics

        echo "Finished inference: $INFER_OUTPUT_DIR"

        cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1
    fi

done

echo "All datasets completed!"
date
