#!/bin/bash
#SBATCH -J det-deepetpicker-czi
#SBATCH -p gpu_pmccaiteam
#SBATCH -t 0-04:00:00
#SBATCH --account=pmccaiteam_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=/cluster/home/t129616uhn/projects/logs/%x_%j.log

# =========================
# DeepETPicker baseline on CZI (prepare -> preprocess -> train -> infer -> score), H4H version of
# baselines/deepetpicker_kit/submit.sbatch.
#   sbatch --export=ALL,PERCENT=100 slurm_scripts_detection/deepetpicker_czi.sh   (100 | 50 | 10)
#
# NOT READY YET: baselines/deepetpicker_kit/setup.sh was written for an Alliance cluster
# (`module load python/3.11`, `pip install --no-index` from the Alliance wheelhouse, torch 2.13).
# $VENV and $DEP below must be built on H4H first; this script refuses to run until they exist.
# BYU is not supported by the kit (hardcoded 6-class CZI config).
# =========================
set -euo pipefail
: "${PERCENT:?set PERCENT}"

KIT="/cluster/home/t129616uhn/projects/CryoDINO/baselines/deepetpicker_kit"
export VENV="${VENV:-/cluster/projects/pmccaiteam/reza/projects/cryoet/envs/deepetpicker}"
export DEP="${DEP:-/cluster/projects/pmccaiteam/reza/projects/cryoet/envs/DeepETPicker}"
export EPOCHS="${EPOCHS:-60}"

[ -f "$KIT/run_datalist.sh" ] || { echo "ERROR: no run_datalist.sh under $KIT"; exit 1; }
[ -f "$VENV/bin/activate" ] || { echo "ERROR: DeepETPicker venv not built at $VENV (see header)"; exit 1; }
[ -d "$DEP" ] || { echo "ERROR: DeepETPicker checkout missing at $DEP (see header)"; exit 1; }

PATCH_DIR="/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets/downstream_detection/Dataset440_CZII_10440_detection_patches_128"
case "$PERCENT" in
  100)   DATALIST="${PATCH_DIR}/czi_100_datalist.json" ;;
  50|10) DATALIST="${PATCH_DIR}/czi_${PERCENT}percent/czi_100_datalist.json" ;;
  *) echo "Unknown PERCENT: $PERCENT"; exit 1 ;;
esac
OUT="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/baselines_out/deepetpicker_czi_${PERCENT}pct"

nvidia-smi -L || true
bash "$KIT/run_datalist.sh" "$DATALIST" "$OUT"
