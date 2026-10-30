#!/bin/bash
#SBATCH -J cryodino-baseline-ds001-rerun-temp
#SBATCH -p gpu_pmccaiteam
#SBATCH -t 4-00:00:00
#SBATCH --account=pmccaiteam_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=800G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/projects/pmccaiteam/reza/projects/cryoet/slurm_logs/%x_%j.log

# =========================
# TEMPORARY rerun script -- delete after use.
# Dataset001 (unet + unetr, 10% and 50%) kept hitting the memory limit in
# train_baselines_patches_10pct.sh and _50pct.sh.
# Round 1 (--mem=440G): the 10% runs finished and are valid (unet 0.8368, unetr 0.7718), but both
# 50% runs were killed ("DataLoader worker killed by signal") about 30% into training and scored a
# truncated checkpoint. sstat showed MaxRSS 466,655,956 KB (~445 GiB), i.e. right at the 440G limit.
# Round 2 (this script): only the two 50% runs, --mem=800G, --num_workers 5 (was 10; memory is
# assumed to scale with the worker count -- unverified), and node164 excluded (a GPU there returns
# "CUDA-capable device(s) is/are busy or unavailable"). Watch MaxRSS with:
#   sstat -j <jobid>.batch --format=JobID,MaxRSS,MaxRSSNode
# If it is still climbing late in the run, suspect a leak rather than a size problem.
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
PERCENTS=("50")   # the 10% runs are done and valid

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
            --num_workers 7 \
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
