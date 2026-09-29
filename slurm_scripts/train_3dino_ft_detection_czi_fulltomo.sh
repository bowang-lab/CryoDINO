#!/bin/bash
#SBATCH -J 3dino-ft-detection-czi-fulltomo
#SBATCH -p gpu_pmcc_ai_team
#SBATCH -t 3-00:00:00
#SBATCH --account=pmcc_ai_team_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

# =========================
# Fine-tuning — DETECTION (CZI), val/test on FULL tomograms. See DETECTION.md.
#
# Usage:
#   sbatch slurm_scripts/train_3dino_ft_detection_czi_fulltomo.sh /path/to/datalist.json [OUTPUT_DIR]
#
#   datalist.json: training = pre-extracted 128^3 .pt patches (local XYZ points);
#                  validation/test = full <run>_0000.nii.gz tomograms (global XYZ points).
#                  Build it with preprocessing/make_detection_split.py --tomo-dir --csv.
#
# Hyperparameters can be overridden from the environment, e.g.
#   EPOCHS=10 EPOCH_LENGTH=202 EVAL_ITERS=505 WARMUP_ITERS=200 sbatch ... <json>
#
# Check a datalist without training (no GPU needed):
#   PREFLIGHT_ONLY=1 bash slurm_scripts/train_3dino_ft_detection_czi_fulltomo.sh <json>
# =========================

DATALIST_JSON="$1"
if [ -z "$DATALIST_JSON" ]; then
  echo "usage: sbatch $0 /path/to/datalist.json [OUTPUT_DIR]" >&2
  exit 1
fi
DATALIST_JSON="$(realpath "$DATALIST_JSON")"

# =========================
# Pre-flight: fail fast on a bad datalist instead of after the first 1500 training iters
# =========================
preflight() {
  python - "$DATALIST_JSON" <<'EOF'
import json, os, sys
path = sys.argv[1]
if not os.path.isfile(path):
    sys.exit(f"PREFLIGHT FAIL: datalist not found: {path}")
d = json.load(open(path))
for k in ("training", "validation", "test"):
    if not d.get(k):
        sys.exit(f"PREFLIGHT FAIL: split {k!r} missing or empty")
if not os.path.isfile(d["training"][0]["image"]):
    sys.exit(f"PREFLIGHT FAIL: training patch not found: {d['training'][0]['image']}  "
             "(regenerate the json for this machine with make_detection_split.py --images-dir)")
for k in ("validation", "test"):
    for e in d[k]:
        img = e["image"]
        if not img.endswith((".nii", ".nii.gz")):
            sys.exit(f"PREFLIGHT FAIL: {k} entry is not a full tomogram (.nii.gz): {img}")
        if not os.path.isfile(img):
            sys.exit(f"PREFLIGHT FAIL: {k} tomogram not found: {img}")
        if not any(p[3] >= 0 for p in e["points"]):
            sys.exit(f"PREFLIGHT FAIL: {k} tomogram has no GT points: {img}")
n = lambda k: sum(sum(p[3] >= 0 for p in e["points"]) for e in d[k])
print(f"PREFLIGHT OK: {len(d['training'])} train patches | "
      f"val {len(d['validation'])} tomo(s), {n('validation')} GT | "
      f"test {len(d['test'])} tomo(s), {n('test')} GT")
EOF
}

if [ -n "$PREFLIGHT_ONLY" ]; then
  preflight
  exit $?
fi

date
hostname
nvidia-smi

source ~/.bashrc
conda activate cryodino

cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1

preflight || exit 1

# =========================
# Paths
# =========================
CONFIG_FILE="dinov2/configs/train/vit3d_highres.yaml"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_b200_high_res_128/eval/training_6249/teacher_checkpoint.pth}"
BASE_OUTPUT_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/finetuning_detection"
CACHE_DIR_BASE="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/cache_dir_downstream_detection"

JSON_NAME="$(basename "$DATALIST_JSON" .json)"
OUTPUT_DIR="${2:-${BASE_OUTPUT_DIR}/czi_detection_fulltomo_${JSON_NAME}_${SLURM_JOB_ID:-local}}"
CACHE_DIR="${CACHE_DIR_BASE}/czi_detection_fulltomo_${JSON_NAME}_${SLURM_JOB_ID:-local}"

# =========================
# Hyperparameters (env-overridable; defaults = train_3dino_ft_h100_detection_czi.sh)
# =========================
DATASET_NAME="czi"            # loaders.py: czi -> 6 classes
DATASET_PERCENT="${DATASET_PERCENT:-100}"
SEGMENTATION_HEAD="${SEGMENTATION_HEAD:-ViTAdapterUNETR}"
EPOCHS="${EPOCHS:-100}"
EPOCH_LENGTH="${EPOCH_LENGTH:-300}"
EVAL_ITERS="${EVAL_ITERS:-1500}"
WARMUP_ITERS="${WARMUP_ITERS:-3000}"
IMAGE_SIZE=128                # must equal the training patch size
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

echo "=============================================="
echo "3D detection fine-tuning (full-tomogram val/test)"
echo "Datalist          : $DATALIST_JSON"
echo "Pretrained weights: $PRETRAINED_WEIGHTS"
echo "Output directory  : $OUTPUT_DIR"
echo "Iters             : ${EPOCHS} x ${EPOCH_LENGTH}, eval every ${EVAL_ITERS}"
echo "=============================================="

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python dinov2/eval/detection3d.py \
  --config-file "$CONFIG_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --pretrained-weights "$PRETRAINED_WEIGHTS" \
  --dataset-name "$DATASET_NAME" \
  --dataset-percent "$DATASET_PERCENT" \
  --base-data-dir "$(dirname "$DATALIST_JSON")" \
  --datalist-json "$DATALIST_JSON" \
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
  --resize-scale 1.0
STATUS=$?

echo "Finished (exit $STATUS): $OUTPUT_DIR"
echo "  results.json   : $OUTPUT_DIR/results.json  (val F4 per eval, test F4 + per-class)"
echo "  best checkpoint: $OUTPUT_DIR/best_model.pth"
date
exit $STATUS
