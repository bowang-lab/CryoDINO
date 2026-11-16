#!/bin/bash
# H4H job array: score ONE periodic checkpoint (task i = line i+1 of $OUT/periodic_checkpoints.txt)
# on the full validation split with CZIDetectionMetrics. Submit after submit_h4h_bench_train.sbatch:
#   sbatch --dependency=afterok:<train jobid> --array=0-19%4 submit_h4h_bench_ckpt.sbatch
# (%4 = at most 4 GPUs at once). Each task has its own predictions/ and infer_work/ folder, and
# resumes per tomogram if resubmitted: sbatch --array=<failed ids> submit_h4h_bench_ckpt.sbatch
#SBATCH --job-name=dep_bench_ckpt
#SBATCH -p gpu_pmccaiteam
#SBATCH --account=pmccaiteam_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/home/%u/deepetpicker/logs/%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail

KIT="${KIT:-$HOME/CryoDINO/baselines/deepetpicker_kit}"   # not $0: sbatch runs a spooled copy
[ -f "$KIT/run_benchmark.sh" ] || { echo "ERROR: no run_benchmark.sh under KIT=$KIT"; exit 1; }
export VENV="${VENV:-$HOME/venvs/deepetpicker}"
export DEP="${DEP:-$HOME/deepetpicker/DeepETPicker}"

: "${DATASET:?set DATASET=czi|byu (--export=ALL,DATASET=...,DATALIST=...,OUT=...)}"
: "${DATALIST:?set DATALIST}"
: "${OUT:?set OUT}"
export DATASET

LIST="$OUT/periodic_checkpoints.txt"
[ -f "$LIST" ] || { echo "ERROR: $LIST missing -- run submit_h4h_bench_train.sbatch (SAVE_EVERY>0) first"; exit 1; }
CKPT="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")"
[ -n "$CKPT" ] || { echo "ERROR: no line $((SLURM_ARRAY_TASK_ID + 1)) in $LIST ($(wc -l < "$LIST") checkpoints)"; exit 1; }
export CKPT STAGE=eval SPLITS=validation
export CHUNK="${CHUNK:-4}"   # 4 tomograms (~1.2 GB of .mrc) at a time per task, so 4 parallel tasks stay ~5 GB

echo "task $SLURM_ARRAY_TASK_ID (attempt $((${SLURM_RESTART_COUNT:-0} + 1))) on $(hostname): $CKPT"

# A broken GPU (seen on node164: "CUDA-capable device(s) is/are busy or unavailable") makes every
# task Slurm hands it fail within seconds, one after another. If ours is broken, keep holding it for
# HOLD_BAD_GPU seconds -- so Slurm cannot give it to the next task -- then requeue this task, which
# goes back to pending and normally lands on a working GPU.
if ! GPU_ERR="$("$VENV/bin/python" -c "import torch; torch.zeros(1, device='cuda'); torch.cuda.synchronize()" 2>&1)"; then
  echo "$GPU_ERR" | tail -2
  echo "GPU ${CUDA_VISIBLE_DEVICES:-?} on $(hostname) is unusable -- holding it ${HOLD_BAD_GPU:-600}s, then requeueing"
  sleep "${HOLD_BAD_GPU:-600}"
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" && exit 0
  echo "ERROR: requeue refused -- resubmit this task: sbatch --array=$SLURM_ARRAY_TASK_ID submit_h4h_bench_ckpt.sbatch"
  exit 1
fi

bash "$KIT/run_benchmark.sh" "$DATALIST" "$OUT"
