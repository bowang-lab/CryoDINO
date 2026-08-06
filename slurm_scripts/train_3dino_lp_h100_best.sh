#!/bin/bash
#SBATCH -J cryodino_lp-h100-best
#SBATCH -p gpu_pmcc_ai_team
#SBATCH -t 7-00:00:00
#SBATCH --account=pmcc_ai_team_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=440G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

# =========================
# H100 linear probing, pinned to its KNOWN BEST checkpoint (training_9374, the
# 112^3 high-res Dice peak) across ALL 4 downstream datasets.
# Uses its OWN cache namespace (lp_h100best_*) — NOT the shared
# ssl3d_run_h100_high_res_training_9374_* dirs the concurrently-running B200
# per-dataset jobs (ds001/ds010/ds989/ds049) are actively reading/writing right
# now. Sharing would risk the exact cross-job cache deadlock already hit twice
# on Dataset001. One-time cache rebuild cost, zero contention risk.
# Results land in the SAME linear_probing_h100_b200_comparison dir for unified
# plotting later.
# =========================

date; hostname; pwd; nvidia-smi

source ~/.bashrc
conda activate cryodino

cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1

EXP="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments"
CONFIG_FILE="dinov2/configs/train/vit3d_highres_112.yaml"
PRETRAINED_WEIGHTS="${EXP}/ssl3d_run_h100_high_res/eval/training_9374/teacher_checkpoint.pth"
IMAGE_SIZE=112
TAG="training_9374"
LABEL="h100_highres"

downstream_datasets=(
    "Dataset001_CZII_10001_patches512"
    "Dataset010_CZII_10010_patches512"
    "Dataset989_EMPIAR_10989_transposed_patches512"
    "Dataset049_EMPIAR_12049_transposed_patches512"
)

# =========================
# Fixed parameters (match the B200 sweep for comparability)
# =========================
BASE_OUTPUT_DIR="${EXP}/linear_probing_h100_b200_comparison"
DATASET_PERCENT=100
BASE_DATA_DIR="${EXP}"
SEGMENTATION_HEAD="Linear"
EPOCHS=100
EPOCH_LENGTH=125
EVAL_ITERS=600
WARMUP_ITERS=3000
BATCH_SIZE=2
NUM_WORKERS=16
LEARNING_RATE=0.001
CACHE_DIR_BASE="${EXP}/cache_dir_downstream"
RESIZE_SCALE=1.0
RUN_LOG_DIR="${BASE_OUTPUT_DIR}/run_logs"
mkdir -p "$BASE_OUTPUT_DIR" "$RUN_LOG_DIR"

if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
    echo "[FATAL] checkpoint not found: $PRETRAINED_WEIGHTS"
    exit 1
fi

for DATASET_NAME in "${downstream_datasets[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${LABEL}_${TAG}_${DATASET_NAME}"
    # Own cache namespace — isolated from the shared per-dataset caches the
    # concurrently-running B200 jobs are using.
    CACHE_DIR="${CACHE_DIR_BASE}/lp_h100best_${DATASET_NAME}"
    if [[ "$DATASET_NAME" == *"12049"* ]]; then
        CACHE_DIR="${CACHE_DIR}_merged"
    fi
    RUN_LOG="${RUN_LOG_DIR}/${LABEL}_${TAG}_${DATASET_NAME}.log"

    if [ -f "${OUTPUT_DIR}/results.json" ] && [ -f "${OUTPUT_DIR}/best_model.pth" ]; then
        echo "  [SKIP done] ${LABEL}/${TAG}/${DATASET_NAME}"
        continue
    fi
    mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

    echo "=============================================="
    echo "Backbone: $LABEL | ckpt: $TAG | image_size: $IMAGE_SIZE"
    echo "Weights:  $PRETRAINED_WEIGHTS"
    echo "Dataset:  $DATASET_NAME  ->  $OUTPUT_DIR"
    echo "=============================================="

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. \
    python dinov2/eval/segmentation3d.py \
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
      --resize-scale "$RESIZE_SCALE" 2>&1 | tee "$RUN_LOG"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "  [FAILED] ${LABEL}/${TAG}/${DATASET_NAME} — continuing"
        continue
    fi
    echo "Finished: $OUTPUT_DIR"
done

echo "H100 best-checkpoint sweep completed!"
date
