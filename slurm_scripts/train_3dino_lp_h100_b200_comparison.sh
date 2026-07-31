#!/bin/bash
#SBATCH -J cryodino_3dino-lp-h100-b200-cmp
#SBATCH -p gpu_pmcc_ai_team
#SBATCH -t 7-00:00:00
#SBATCH --account=pmcc_ai_team_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

# =========================
# Linear-probing comparison across ALL eval checkpoints of each run:
#   H100  pretraining + high-res adaptation
#   B200  pretraining + high-res 112 + high-res 128
#   random init (no pretraining) baseline
# Frozen ViT + Linear head. Sweeps every eval/training_*/teacher_checkpoint.pth
# so you can plot downstream Dice vs pretraining iteration.
#
# Large sweep -> two built-in controls:
#   * RESUME: a run whose results.json already exists is skipped, so re-submitting
#     (or a second 7-day job) continues where the last one stopped.
#   * SPLIT: pass run labels as arguments to restrict this job to a subset, e.g.
#       sbatch train_3dino_lp_h100_b200_comparison.sh h100_pretrain b200_pretrain
#     With no args, all runs are processed.
# Each run writes results.json + best_model.pth to its OUTPUT_DIR (plot from those).
# =========================

date; hostname; pwd; nvidia-smi

source ~/.bashrc
conda activate cryodino

cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1

CONFIG_DEFAULT="dinov2/configs/ssl3d_default_config.yaml"
CONFIG_HIGHRES="dinov2/configs/train/vit3d_highres.yaml"
EXP="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments"

# Runs:  label | run_dir (glob eval/training_*/teacher_checkpoint.pth) | config | image_size
# random_init has no run_dir (handled specially: single run, empty weights).
RUNS=(
    "h100_pretrain|${EXP}/ssl3d_run_h100|${CONFIG_DEFAULT}|96"
    "h100_highres|${EXP}/ssl3d_run_h100_high_res|${CONFIG_HIGHRES}|112"
    "b200_pretrain|${EXP}/ssl3d_run_b200|${CONFIG_DEFAULT}|96"
    "b200_highres112|${EXP}/ssl3d_run_b200_high_res|${CONFIG_HIGHRES}|112"
    "b200_highres128|${EXP}/ssl3d_run_b200_high_res_128|${CONFIG_HIGHRES}|128"
    "random_init|RANDOM|${CONFIG_HIGHRES}|112"
)

downstream_datasets=(
    "Dataset001_CZII_10001_patches512"
    "Dataset010_CZII_10010_patches512"
    "Dataset989_EMPIAR_10989_transposed_patches512"
    "Dataset049_EMPIAR_12049_transposed_patches512"
)

# Optional job-splitting: restrict to labels passed as args (default = all)
SELECT=("$@")
selected() {
    [ ${#SELECT[@]} -eq 0 ] && return 0
    for s in "${SELECT[@]}"; do [ "$s" == "$1" ] && return 0; done
    return 1
}

# =========================
# Fixed parameters (match existing linear-probing jobs for comparability)
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

run_one() {
    local LABEL="$1" WEIGHTS="$2" CONFIG_FILE="$3" IMAGE_SIZE="$4" TAG="$5" DATASET_NAME="$6"
    local OUTPUT_DIR="${BASE_OUTPUT_DIR}/${LABEL}_${TAG}_${DATASET_NAME}"
    # Reuse existing per-dataset cache (cache stores pre-crop ~512^3 volume, keyed
    # by data item, so it is shared across all backbones/image sizes).
    # 12049 uses the "_merged" cache: its label remap (_remap_12049) lives in the
    # cached load_transforms, so the merged-label cache is a separate dir.
    local CACHE_DIR="${CACHE_DIR_BASE}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}"
    if [[ "$DATASET_NAME" == *"12049"* ]]; then
        CACHE_DIR="${CACHE_DIR}_merged"
    fi
    local RUN_LOG="${RUN_LOG_DIR}/${LABEL}_${TAG}_${DATASET_NAME}.log"

    if [ -f "${OUTPUT_DIR}/results.json" ]; then
        echo "  [SKIP done] ${LABEL}/${TAG}/${DATASET_NAME}"
        return 0
    fi
    mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

    echo "=============================================="
    echo "Backbone: $LABEL | ckpt: $TAG | image_size: $IMAGE_SIZE"
    echo "Weights:  ${WEIGHTS:-<random init>}"
    echo "Dataset:  $DATASET_NAME  ->  $OUTPUT_DIR"
    echo "=============================================="

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. \
    python dinov2/eval/segmentation3d.py \
      --config-file "$CONFIG_FILE" \
      --output-dir "$OUTPUT_DIR" \
      --pretrained-weights "$WEIGHTS" \
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
      --resize-scale "$RESIZE_SCALE" 2>&1 | tee "$RUN_LOG" \
      || { echo "  [FAILED] ${LABEL}/${TAG}/${DATASET_NAME} — continuing"; return 0; }
    echo "Finished: $OUTPUT_DIR"
}

for R in "${RUNS[@]}"; do
    IFS='|' read -r LABEL RUN_DIR CONFIG_FILE IMAGE_SIZE <<< "$R"
    selected "$LABEL" || { echo "Skipping $LABEL (not in selection)"; continue; }

    if [ "$RUN_DIR" == "RANDOM" ]; then
        for DS in "${downstream_datasets[@]}"; do
            run_one "$LABEL" "" "$CONFIG_FILE" "$IMAGE_SIZE" "randominit" "$DS"
        done
        continue
    fi

    # discover every eval checkpoint for this run (sorted by iteration)
    CKPTS=( $(ls -1 "${RUN_DIR}"/eval/training_*/teacher_checkpoint.pth 2>/dev/null | sort -t_ -k2 -n) )
    if [ ${#CKPTS[@]} -eq 0 ]; then
        echo "  [WARN] no checkpoints found under ${RUN_DIR}/eval/ — skipping $LABEL"
        continue
    fi
    echo ">>> $LABEL: ${#CKPTS[@]} checkpoints found"

    for CKPT in "${CKPTS[@]}"; do
        TAG=$(basename "$(dirname "$CKPT")")   # e.g. training_124999
        for DS in "${downstream_datasets[@]}"; do
            run_one "$LABEL" "$CKPT" "$CONFIG_FILE" "$IMAGE_SIZE" "$TAG" "$DS"
        done
    done
done

echo "All requested linear-probing runs completed!"
date
