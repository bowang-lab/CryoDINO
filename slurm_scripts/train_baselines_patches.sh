#!/bin/bash
#SBATCH -J cryodino-baseline
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
# Baseline training: UNet and UNETR from random initialization
# All 4 datasets, 100% data, 200 epochs (60k iterations)
# Runs sequentially: unet × 4 datasets → unetr × 4 datasets (8 runs total)
# Submit: sbatch train_baselines_patches.sh
# =========================

date
hostname
pwd
nvidia-smi

source ~/.bashrc
conda activate cryoet

BASE_DATA_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments"
CACHE_DIR_BASE="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/cache_dir_downstream"
OUTPUT_BASE="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/baselines"
IMAGE_SIZE=112
BATCH_SIZE=2

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
NUM_CLASSES["Dataset049_EMPIAR_12049_transposed_patches512"]=6

DATASETS=(
    "Dataset001_CZII_10001_patches512"
    "Dataset010_CZII_10010_patches512"
    "Dataset989_EMPIAR_10989_transposed_patches512"
    "Dataset049_EMPIAR_12049_transposed_patches512"
)
MODELS=("unet" "unetr")

cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1

for MODEL in "${MODELS[@]}"; do
    for DATASET_NAME in "${DATASETS[@]}"; do

        CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}"
        OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}_${DATASET_NAME}"

        mkdir -p "$OUTPUT_DIR"
        mkdir -p "$CACHE_DIR"

        echo "=================================================="
        echo "  Model   : ${MODEL}"
        echo "  Dataset : ${DATASET_NAME}"
        echo "  Output  : ${OUTPUT_DIR}"
        echo "  Cache   : ${CACHE_DIR}"
        echo "=================================================="

        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python ../baselines/train.py \
            --model_name "${MODEL}" \
            --dataset_name "${DATASET_NAME}" \
            --base_data_dir "${BASE_DATA_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --cache_dir "${CACHE_DIR}" \
            --image_size "$IMAGE_SIZE" \
            --epochs 100 \
            --epoch_length 300 \
            --eval_iters 600 \
            --warmup_iters 3000 \
            --learning_rate 1e-4 \
            --batch_size "$BATCH_SIZE" \
            --num_workers 10 \
            --dataset_percent 100

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
            echo "Running inference: ${MODEL} on ${DATASET_NAME}"
            echo "Checkpoint: $CHECKPOINT"
            echo "Output:     $INFER_OUTPUT_DIR"
            echo "----------------------------------------------"

            OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python ../baselines/inference.py \
              --model-name "$MODEL" \
              --checkpoint "$CHECKPOINT" \
              --image-size "$IMAGE_SIZE" \
              --num-classes "${NUM_CLASSES[$DATASET_NAME]}" \
              --datalist "${DATALIST[$DATASET_NAME]}" \
              --dataset-name "$DATASET_NAME" \
              --output-dir "$INFER_OUTPUT_DIR" \
              --batch-size "$BATCH_SIZE" \
              --cpu-metrics

            echo "Finished inference: $INFER_OUTPUT_DIR"
        fi

    done
done

echo "All datasets completed!"
date
