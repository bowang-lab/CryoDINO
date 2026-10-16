#!/bin/bash
# Short training run whose only purpose is to show the learning curves (train/val loss down,
# val metrics up).  10 epochs is enough to see a slope, and the small walltime request makes the
# job backfill-eligible instead of queueing behind a multi-hour reservation.
#   [EPOCHS=10] sbatch curves.sbatch
#SBATCH --job-name=dep_curves
#SBATCH --account=rrg-wanglab-ab
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/cyyu/cryodino/logs/%x-%j.out
set -uo pipefail

KIT="${KIT:-/home/cyyu/projects/def-wanglab-ab/cyyu/deepetpicker_kit}"
[ -f "$KIT/run_datalist.sh" ] || { echo "ERROR: no run_datalist.sh under KIT=$KIT"; exit 1; }
export VENV="${VENV:-/scratch/cyyu/venvs/deepetpicker}"
export DEP="${DEP:-/scratch/cyyu/cryodino/DeepETPicker}"
DATALIST="${DATALIST:-/home/cyyu/projects/def-wanglab-ab/cryodino/Dataset440_CZII_10440_detection_patches_128/czi_100_datalist.json}"
OUT="${OUT:-/scratch/cyyu/cryodino/deepetpicker_ds440}"

module load python/3.11 2>/dev/null || true
nvidia-smi -L
EPOCHS="${EPOCHS:-10}" bash "$KIT/run_datalist.sh" "$DATALIST" "$OUT"
echo "CURVES exit=$?"
