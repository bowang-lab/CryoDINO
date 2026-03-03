#!/bin/bash
#SBATCH -J 3dino-inference-all
#SBATCH -p gpu_bwanggroup
#SBATCH -t 3-00:00:00
#SBATCH --account=bwanggroup_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=300G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

# =========================
# Batch inference for all fine-tuning experiments (patches512 mode)
# Runs sliding window inference + Dice/HD95 on the test split of each dataset.
# Segmentation head (UNETR vs ViTAdapterUNETR) and dataset config are
# auto-detected from the experiment directory name.
# Output: {EXP_DIR}/inference/  (predictions + metrics.json)
# =========================

date
hostname
pwd
nvidia-smi

source ~/.bashrc
conda activate cryoet

cd /cluster/home/t139212uhn/scripts/cryoet/CryoET || exit 1

# =========================
# Fixed Parameters
# =========================
CONFIG_FILE="3DINO/dinov2/configs/train/vit3d_highres.yaml"
PRETRAINED_WEIGHTS="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100_high_res/eval/training_9374/teacher_checkpoint.pth"
FINETUNING_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/finetuning"
DATA_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments"
IMAGE_SIZE=112
BATCH_SIZE=2       # keep low to leave headroom for CPU-side metric computation
OVERLAP=0.75

# Datalist for each dataset (test split is identical across 10/50/100% variants)
declare -A DATALIST
DATALIST["10001"]="${DATA_DIR}/Dataset001_CZII_10001_patches512_100_datalist.json"
DATALIST["10010"]="${DATA_DIR}/Dataset010_CZII_10010_patches512_100_datalist.json"
# DATALIST["12049"]="${DATA_DIR}/Dataset049_EMPIAR_12049_transposed_patches512_100_datalist.json"
DATALIST["10989"]="${DATA_DIR}/Dataset989_EMPIAR_10989_transposed_patches512_100_datalist.json"

# Number of segmentation classes per dataset
declare -A NUM_CLASSES
NUM_CLASSES["10001"]=4
NUM_CLASSES["10010"]=2
# NUM_CLASSES["12049"]=4
NUM_CLASSES["10989"]=2

# Dataset name passed to --dataset-name (used by build_transforms for label handling)
declare -A DATASET_NAME
DATASET_NAME["10001"]="Dataset001_CZII_10001"
DATASET_NAME["10010"]="Dataset010_CZII_10010"
# DATASET_NAME["12049"]="Dataset049_EMPIAR_12049_transposed"
DATASET_NAME["10989"]="Dataset989_EMPIAR_10989_transposed"

# All experiment directories to run inference on
EXPERIMENTS=(
    # ---- Dataset001 (CZII 10001, 4-class) ----
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512"
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512_vit_adapter"
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512_10percent"
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512_10percent_vit_adapter"
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512_50percent"
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512_50percent_vit_adapter"
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512_full_ft"
    "ssl3d_run_h100_high_res_training_9374_Dataset001_CZII_10001_patches512_full_ft_vit_adapter"
    # ---- Dataset010 (CZII 10010, binary) ----
    "ssl3d_run_h100_high_res_training_9374_Dataset010_CZII_10010_patches512"
    "ssl3d_run_h100_high_res_training_9374_Dataset010_CZII_10010_patches512_vit_adapter"
    "ssl3d_run_h100_high_res_training_9374_Dataset010_CZII_10010_patches512_10percent"
    "ssl3d_run_h100_high_res_training_9374_Dataset010_CZII_10010_patches512_10percent_vit_adapter"
    "ssl3d_run_h100_high_res_training_9374_Dataset010_CZII_10010_patches512_50percent"
    "ssl3d_run_h100_high_res_training_9374_Dataset010_CZII_10010_patches512_50percent_vit_adapter"
    "ssl3d_run_h100_high_res_training_9374_Dataset010_CZII_10010_patches512_full_ft"
    "ssl3d_run_h100_high_res_training_9374_Dataset010_CZII_10010_patches512_full_ft_vit_adapter"
    # ---- Dataset049 (EMPIAR 12049, 4-class after remap) ----
    # "ssl3d_run_h100_high_res_training_9374_Dataset049_EMPIAR_12049_transposed_patches512"
    # "ssl3d_run_h100_high_res_training_9374_Dataset049_EMPIAR_12049_transposed_patches512_vit_adapter"
    # "ssl3d_run_h100_high_res_training_9374_Dataset049_EMPIAR_12049_transposed_patches512_vit_adapter2"
    # ---- Dataset989 (EMPIAR 10989, binary) ----
    "ssl3d_run_h100_high_res_training_9374_Dataset989_EMPIAR_10989_transposed_patches512"
    "ssl3d_run_h100_high_res_training_9374_Dataset989_EMPIAR_10989_transposed_patches512_vit_adapter"
    "ssl3d_run_h100_high_res_training_9374_Dataset989_EMPIAR_10989_transposed_patches512_10percent"
    "ssl3d_run_h100_high_res_training_9374_Dataset989_EMPIAR_10989_transposed_patches512_10percent_vit_adapter"
    "ssl3d_run_h100_high_res_training_9374_Dataset989_EMPIAR_10989_transposed_patches512_50percent"
    "ssl3d_run_h100_high_res_training_9374_Dataset989_EMPIAR_10989_transposed_patches512_50percent_vit_adapter"
    "ssl3d_run_h100_high_res_training_9374_Dataset989_EMPIAR_10989_transposed_patches512_full_ft"
    "ssl3d_run_h100_high_res_training_9374_Dataset989_EMPIAR_10989_transposed_patches512_full_ft_vit_adapter"
)

SKIPPED=()
FAILED=()

for EXP in "${EXPERIMENTS[@]}"; do

    echo ""
    echo "=============================================="
    echo "Experiment: $EXP"
    echo "=============================================="

    # -- Checkpoint --
    CHECKPOINT="${FINETUNING_DIR}/${EXP}/best_model.pth"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "  [SKIP] checkpoint not found: $CHECKPOINT"
        SKIPPED+=("$EXP")
        continue
    fi

    # -- Segmentation head: ViTAdapterUNETR if name contains "_vit_adapter" --
    if [[ "$EXP" == *"_vit_adapter"* ]]; then
        SEG_HEAD="ViTAdapterUNETR"
    else
        SEG_HEAD="UNETR"
    fi

    # -- Dataset ID from experiment name --
    if [[ "$EXP" == *"10001"* ]]; then
        DATASET_ID="10001"
    elif [[ "$EXP" == *"10010"* ]]; then
        DATASET_ID="10010"
    elif [[ "$EXP" == *"12049"* ]]; then
        DATASET_ID="12049"
    elif [[ "$EXP" == *"10989"* ]]; then
        DATASET_ID="10989"
    else
        echo "  [SKIP] cannot determine dataset ID from: $EXP"
        SKIPPED+=("$EXP")
        continue
    fi

    DATALIST_FILE="${DATALIST[$DATASET_ID]}"
    CLASSES="${NUM_CLASSES[$DATASET_ID]}"
    DS_NAME="${DATASET_NAME[$DATASET_ID]}"
    OUTPUT_DIR="${FINETUNING_DIR}/${EXP}/inference"
    mkdir -p "$OUTPUT_DIR"

    echo "  Head:       $SEG_HEAD"
    echo "  Dataset:    $DS_NAME  (classes=$CLASSES)"
    echo "  Datalist:   $DATALIST_FILE"
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Output:     $OUTPUT_DIR"

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python inference/segmentation3d_inference.py \
        --config-file "$CONFIG_FILE" \
        --pretrained-weights "$PRETRAINED_WEIGHTS" \
        --checkpoint "$CHECKPOINT" \
        --segmentation-head "$SEG_HEAD" \
        --image-size "$IMAGE_SIZE" \
        --num-classes "$CLASSES" \
        --datalist "$DATALIST_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --dataset-name "$DS_NAME" \
        --overlap "$OVERLAP" \
        --batch-size "$BATCH_SIZE"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "  [FAILED] exit code $EXIT_CODE"
        FAILED+=("$EXP")
    else
        echo "  [DONE]"
    fi

done

# Summary
echo ""
echo "=============================================="
echo "INFERENCE SUMMARY"
echo "=============================================="
echo "Skipped (no checkpoint): ${#SKIPPED[@]}"
for s in "${SKIPPED[@]}"; do echo "  - $s"; done
echo "Failed: ${#FAILED[@]}"
for f in "${FAILED[@]}"; do echo "  - $f"; done
echo "Completed: $((${#EXPERIMENTS[@]} - ${#SKIPPED[@]} - ${#FAILED[@]})) / ${#EXPERIMENTS[@]}"

date
