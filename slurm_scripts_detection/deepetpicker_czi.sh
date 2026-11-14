#!/bin/bash
#SBATCH -J det-deepetpicker-czi
#SBATCH -p gpu_pmccaiteam
#SBATCH -t 2-00:00:00
#SBATCH --account=pmccaiteam_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/cluster/home/t129616uhn/projects/logs/%x_%j.log

# =========================
# DeepETPicker baseline on CZI, benchmark split (same as train_detection.sh / detection3d.py):
#   train  Dataset440 (the datalist's training patches, reassembled into tomograms)
#   val    Dataset445 (121 tomograms): picks the per-class score thresholds
#   test   Dataset446 (364 tomograms): scored once at those thresholds, CZIDetectionMetrics
#
#   sbatch --export=ALL,PERCENT=100 slurm_scripts_detection/deepetpicker_czi.sh   (100 | 50 | 10)
#
# Env (built on H4H, torch 2.7.1 cu128 so any GPU in the partition works):
#   $VENV  /cluster/projects/pmccaiteam/reza/projects/cryoet/envs/deepetpicker
#   $DEP   /cluster/projects/pmccaiteam/reza/projects/cryoet/envs/DeepETPicker (kit patches applied)
# BYU is not supported by the kit (6-class CZI config).
# =========================
set -euo pipefail
: "${PERCENT:?set PERCENT}"

REPO="/cluster/home/t129616uhn/projects/CryoDINO"
KIT="$REPO/baselines/deepetpicker_kit"
export VENV="${VENV:-/cluster/projects/pmccaiteam/reza/projects/cryoet/envs/deepetpicker}"
export DEP="${DEP:-/cluster/projects/pmccaiteam/reza/projects/cryoet/envs/DeepETPicker}"
export REPO
export EPOCHS="${EPOCHS:-60}"
export WORKERS="${SLURM_CPUS_PER_TASK:-16}"
export ORIG_TRAIN_NII="/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets/downstream_detection/czi_dataset/Dataset440_CZII_10440/imagesTr"

[ -f "$KIT/run_benchmark.sh" ] || { echo "ERROR: no run_benchmark.sh under $KIT"; exit 1; }
[ -f "$VENV/bin/activate" ] || { echo "ERROR: DeepETPicker venv not built at $VENV"; exit 1; }
[ -d "$DEP" ] || { echo "ERROR: DeepETPicker checkout missing at $DEP"; exit 1; }

PATCH_DIR="/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets/downstream_detection/Dataset440_CZII_10440_detection_patches_128"
case "$PERCENT" in
  100)   DATALIST="${PATCH_DIR}/czi_100_datalist.json" ;;
  50|10) DATALIST="${PATCH_DIR}/czi_${PERCENT}percent/czi_100_datalist.json" ;;
  *) echo "Unknown PERCENT: $PERCENT"; exit 1 ;;
esac
OUT="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/baselines_out/deepetpicker_czi_${PERCENT}pct"

date
hostname
nvidia-smi -L || true
echo "Datalist: $DATALIST"
echo "Output:   $OUT"
echo "Git: $(git -C "$REPO" rev-parse --short HEAD) on $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"

bash "$KIT/run_benchmark.sh" "$DATALIST" "$OUT"
date
