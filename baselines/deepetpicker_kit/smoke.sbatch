#!/bin/bash
# 1-epoch end-to-end smoke test: prepare -> preprocess -> train -> infer -> score.
# Proves the patched DeepETPicker still runs on the modern stack before spending a full run.
#   sbatch smoke.sbatch
#SBATCH --job-name=dep_smoke
#SBATCH --account=rrg-wanglab-ab
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:30:00
#SBATCH --output=/scratch/cyyu/cryodino/logs/dep_smoke-%j.out
set -uo pipefail

# NOT $0/BASH_SOURCE: sbatch copies this script to a per-job spool dir on the compute node
# (/localscratch/spool/slurmd/job<ID>/), so deriving the kit path from it finds nothing.
KIT="${KIT:-/home/cyyu/projects/def-wanglab-ab/cyyu/deepetpicker_kit}"
[ -f "$KIT/run_datalist.sh" ] || { echo "ERROR: no run_datalist.sh under KIT=$KIT -- set KIT=<kit dir>"; exit 1; }
export VENV="${VENV:-/scratch/cyyu/venvs/deepetpicker}"
export DEP="${DEP:-/scratch/cyyu/cryodino/DeepETPicker}"
DATALIST="${DATALIST:-/home/cyyu/projects/def-wanglab-ab/cryodino/Dataset440_CZII_10440_detection_patches_128/czi_100_datalist.json}"
OUT="${OUT:-/scratch/cyyu/cryodino/deepetpicker_ds440}"

module load python/3.11 2>/dev/null || true
nvidia-smi -L
EPOCHS=1 bash "$KIT/run_datalist.sh" "$DATALIST" "$OUT"
echo "SMOKE exit=$?"
