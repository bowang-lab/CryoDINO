#!/bin/bash
#SBATCH -J monai-retinanet-detection-czi
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
# Baseline: MONAI RetinaNet 3D (ResNet-FPN, from scratch) — DETECTION (CZI)
# Validated/tested with the CryoDINO metric (CZIDetectionMetrics, Kaggle CZII F4) so it is
# directly comparable to train_3dino_ft_h100_detection_czi.sh.
#
# Key differences vs detection3d.py:
#   * No DINO backbone / config / pretrained weights.
#   * Train on the honest-split 128^3 .pt patches (5 tomograms, split by tomogram).
#   * Val (TS_69_2) and test (TS_73_6) run sliding-window over the FULL tomogram NIfTI with
#     global GT points from point_annotations.csv (each particle counted once).
#   * The honest-split datalist stores absolute patch paths — regenerate it for this cluster
#     with preprocessing/make_detection_split.py if the paths differ.
# =========================

date
hostname
pwd
nvidia-smi

source ~/.bashrc
conda activate cryodino

cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1

# =========================
# Paths
# =========================
DATA_ROOT="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream_detection"
DATALIST="${DATA_ROOT}/Dataset440_CZII_10440_detection_patches_128/honest-split_datalist.json"
IMAGES_DIR="${DATA_ROOT}/Dataset440_CZII_10440/imagesTr"
ANNOTATIONS_CSV="${DATA_ROOT}/Dataset440_CZII_10440/point_annotations.csv"
BASE_OUTPUT_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/finetuning_detection"
CACHE_DIR_BASE="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/cache_dir_downstream_detection"

# =========================
# Fixed Parameters
# =========================
RESNET_DEPTH=34
EPOCHS=100
EPOCH_LENGTH=300
EVAL_ITERS=1500
WARMUP_ITERS=3000
IMAGE_SIZE=128                # = patch size = sliding-window roi
BATCH_SIZE=4
NUM_WORKERS=10
LEARNING_RATE=1e-4

OUTPUT_DIR="${BASE_OUTPUT_DIR}/monai_retinanet_resnet${RESNET_DEPTH}_czi_detection"
CACHE_DIR="${CACHE_DIR_BASE}/monai_retinanet_czi_detection"

rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Starting MONAI RetinaNet detection training..."
echo "Datalist: $DATALIST"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python dinov2/eval/detection3d_monai.py \
  --datalist "$DATALIST" \
  --images-dir "$IMAGES_DIR" \
  --annotations-csv "$ANNOTATIONS_CSV" \
  --val-runs TS_69_2 \
  --test-runs TS_73_6 \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --resnet-depth "$RESNET_DEPTH" \
  --epochs "$EPOCHS" \
  --epoch-length "$EPOCH_LENGTH" \
  --eval-iters "$EVAL_ITERS" \
  --warmup-iters "$WARMUP_ITERS" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --learning-rate "$LEARNING_RATE"

echo "Finished detection training: $OUTPUT_DIR"
echo "  (test F4 threshold-sweep + results.json written by detection3d_monai.py)"
date
