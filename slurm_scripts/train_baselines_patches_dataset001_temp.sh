#!/bin/bash
#SBATCH -J cryodino-baseline-ds001-rerun-temp
#SBATCH -p gpu_pmccaiteam
#SBATCH -t 4-00:00:00
#SBATCH --account=pmccaiteam_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=440G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/projects/pmccaiteam/reza/projects/cryoet/slurm_logs/%x_%j.log

# =========================
# TEMPORARY rerun script -- delete after use.
# Dataset001 (unet + unetr, 10% and 50%) kept hitting --mem=220G OOM in both
# train_baselines_patches_10pct.sh and _50pct.sh (all 4 combos invalid: truncated
# training, and unetr/50pct's inference process was also OOM-killed outright).
# Bumped to --mem=440G here (2x the failing 220G), scoped to just the 4 broken runs,
# so we don't have to re-run the 6 already-valid combos in each job.
# =========================

date
hostname
pwd
nvidia-smi

source ~/.bashrc
conda activate cryodino

BASE_DATA_DIR="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments"
CACHE_DIR_BASE="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/cache_dir_downstream"
OUTPUT_BASE="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/baselines"
IMAGE_SIZE=112
BATCH_SIZE=2
DATASET_PERCENT=100

CACHE_DATASET="Dataset001_CZII_10001_patches512"
NUM_CLASSES=4
DATALIST="${BASE_DATA_DIR}/${CACHE_DATASET}_100_datalist.json"
CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${CACHE_DATASET}"

MODELS=("unet" "unetr")
PERCENTS=("10" "50")

cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1

for PCT in "${PERCENTS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

        DATASET_NAME="${CACHE_DATASET}_${PCT}percent"
        OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}_${DATASET_NAME}"

        mkdir -p "$OUTPUT_DIR"
        mkdir -p "$CACHE_DIR"

        echo "=================================================="
        echo "  Model   : ${MODEL}"
        echo "  Dataset : ${DATASET_NAME}"
        echo "  Output  : ${OUTPUT_DIR}"
        echo "  Cache   : ${CACHE_DIR} (shared with 100% run)"
        echo "=================================================="

        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python ../baselines/unet_unetr/train.py \
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
            --dataset_percent "$DATASET_PERCENT"

        echo "Finished training: $OUTPUT_DIR"

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

            OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python ../baselines/unet_unetr/inference.py \
              --model-name "$MODEL" \
              --checkpoint "$CHECKPOINT" \
              --image-size "$IMAGE_SIZE" \
              --num-classes "$NUM_CLASSES" \
              --datalist "$DATALIST" \
              --dataset-name "$CACHE_DATASET" \
              --output-dir "$INFER_OUTPUT_DIR" \
              --batch-size "$BATCH_SIZE" \
              --cpu-metrics

            echo "Finished inference: $INFER_OUTPUT_DIR"
        fi

    done
done

echo "All Dataset001 reruns completed!"
date
