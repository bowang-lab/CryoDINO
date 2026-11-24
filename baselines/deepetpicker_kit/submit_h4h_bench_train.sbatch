#!/bin/bash
# H4H: train DeepETPicker on every datalist `training` tomogram (100%: all 7 of Dataset440).
# Training budget and protocol defaults (100 epochs x 300 iterations, batch 4, lr 5e-5, checkpoint
# every 5 epochs; BYU block 128 / label 13) live in run_benchmark.sh -- override with --export=ALL,VAR=...
# Periodic checkpoints -> $OUT/periodic_checkpoints.txt,
# scored by submit_h4h_bench_ckpt.sbatch and selected by validation F4 in submit_h4h_bench_select.sbatch.
# DeepETPicker's own best-val_loss checkpoint is still kept -> $OUT/checkpoint_used.txt.
#
#   Chain (train -> score 20 checkpoints on val -> pick best -> test); DATASET, DATALIST, OUT required:
#     E=ALL,DATASET=czi,DATALIST=<datalist.json>,OUT=<out_dir>     (+ REPEAT_TRAIN=2 / 9 for CZI 50% / 10%)
#     t=$(sbatch --parsable --export=$E submit_h4h_bench_train.sbatch)
#     c=$(sbatch --parsable --dependency=afterok:$t --array=0-19%4 --export=$E submit_h4h_bench_ckpt.sbatch)
#     sbatch --dependency=afterok:$c --export=$E submit_h4h_bench_select.sbatch      (BYU: add CHUNK=2)
#   slurm_scripts/deepetpicker_{czi,byu}.sh submit exactly this.
#   smoke: add EPOCHS=1 to E (and LIMIT=4 for the ckpt/select jobs)
#SBATCH --job-name=dep_bench_train
#SBATCH -p gpu_pmccaiteam
#SBATCH --account=pmccaiteam_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_logs/%x-%j.out
set -euo pipefail

KIT="${KIT:-$HOME/CryoDINO/baselines/deepetpicker_kit}"   # not $0: sbatch runs a spooled copy
[ -f "$KIT/run_benchmark.sh" ] || { echo "ERROR: no run_benchmark.sh under KIT=$KIT"; exit 1; }
export VENV="${VENV:-$HOME/venvs/deepetpicker}"
export DEP="${DEP:-$HOME/deepetpicker/DeepETPicker}"
export STAGE=train

: "${DATASET:?set DATASET=czi|byu (--export=ALL,DATASET=...,DATALIST=...,OUT=...)}"
: "${DATALIST:?set DATALIST}"
: "${OUT:?set OUT}"
export DATASET

echo "STAGE=train DATASET=$DATASET OUT=$OUT (settings: see run_benchmark.sh step 5 line)"
bash "$KIT/run_benchmark.sh" "$DATALIST" "$OUT"
