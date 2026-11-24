#!/bin/bash
# H4H, last step: pick the periodic checkpoint with the best validation F4 (select_checkpoint.py),
# then score it on the test split. Validation predictions for it already exist and are reused, so
# metrics_czi.json gets val F4, test F4, and test F4 at the validation-chosen thresholds.
#   sbatch --dependency=afterok:<array jobid> submit_h4h_bench_select.sbatch
#SBATCH --job-name=dep_bench_select
#SBATCH -p gpu_pmccaiteam
#SBATCH --account=pmccaiteam_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/%x-%j.out
set -euo pipefail

KIT="${KIT:-$HOME/CryoDINO/baselines/deepetpicker_kit}"   # not $0: sbatch runs a spooled copy
[ -f "$KIT/run_benchmark.sh" ] || { echo "ERROR: no run_benchmark.sh under KIT=$KIT"; exit 1; }
export VENV="${VENV:-$HOME/venvs/deepetpicker}"
export DEP="${DEP:-$HOME/deepetpicker/DeepETPicker}"

: "${DATASET:?set DATASET=czi|byu (--export=ALL,DATASET=...,DATALIST=...,OUT=...)}"
: "${DATALIST:?set DATALIST}"
: "${OUT:?set OUT}"
export DATASET

source "$VENV/bin/activate"
python "$KIT/select_checkpoint.py" --out "$OUT"
export CKPT="$(cat "$OUT/best_f4_checkpoint.txt")" STAGE=eval SPLITS=validation,test
bash "$KIT/run_benchmark.sh" "$DATALIST" "$OUT"
