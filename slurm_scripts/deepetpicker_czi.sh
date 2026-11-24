#!/bin/bash
# =========================
# DeepETPicker baseline on CZI, benchmark split, same protocol as detection3d.py / train_detection.sh:
#   train  Dataset440 (the datalist's training patches, reassembled into tomograms),
#          100 epochs x 300 iterations = 30k, batch 4, lr 5e-5, a checkpoint every 5 epochs (20)
#   val    Dataset445 (121 tomograms): scores all 20 checkpoints, picks the best one and the
#          per-class thresholds
#   test   Dataset446 (364 tomograms): scored once with that checkpoint at those thresholds
#   metric CZIDetectionMetrics from 3DINO/dinov2/eval/detection_3d/metrics.py
#
# Run on the LOGIN node (it only submits; three dependent jobs: train -> 20-task ckpt array -> select):
#   PERCENT=100 bash slurm_scripts/deepetpicker_czi.sh      (100 | 50 | 10)
# Training defaults live in baselines/deepetpicker_kit/run_benchmark.sh; BYU: DATASET=byu with the
# same kit: slurm_scripts/deepetpicker_byu.sh.
#
# Env (built on H4H, torch 2.7.1 cu128 so any GPU in the partition works):
#   $VENV  /cluster/projects/pmccaiteam/<USERNAME>/projects/cryoet/envs/deepetpicker
#   $DEP   /cluster/projects/pmccaiteam/<USERNAME>/projects/cryoet/envs/DeepETPicker
#          (run_benchmark.sh applies patch_periodic_ckpt.py to it)
# =========================
set -euo pipefail
: "${PERCENT:?set PERCENT (100 | 50 | 10)}"
[ -z "${SLURM_JOB_ID:-}" ] || { echo "ERROR: run this with bash on the login node, not sbatch (it submits the jobs itself)"; exit 1; }

REPO="${REPO:-/cluster/home/$USER/projects/CryoDINO}"
export KIT="$REPO/baselines/deepetpicker_kit"
export VENV="${VENV:-/cluster/projects/pmccaiteam/<USERNAME>/projects/cryoet/envs/deepetpicker}"
export DEP="${DEP:-/cluster/projects/pmccaiteam/<USERNAME>/projects/cryoet/envs/DeepETPicker}"
LOGS="${LOGS:-$PWD/slurm_logs}"   # run from the repo root, like the #SBATCH --output=slurm_logs/ lines

[ -f "$KIT/run_benchmark.sh" ] || { echo "ERROR: no run_benchmark.sh under $KIT"; exit 1; }
[ -f "$VENV/bin/activate" ] || { echo "ERROR: DeepETPicker venv not built at $VENV"; exit 1; }
[ -d "$DEP" ] || { echo "ERROR: DeepETPicker checkout missing at $DEP"; exit 1; }
mkdir -p "$LOGS"

PATCH_DIR="/cluster/projects/pmccaiteam/<USERNAME>/projects/cryoet/datasets/downstream_detection/Dataset440_CZII_10440_detection_patches_128"
# REPEAT_TRAIN: the subsets have too few training samples for 300 batches of 4 per epoch, so an epoch
# shuffles over N copies of them (the values used for the reported runs)
case "$PERCENT" in
  100) DATALIST="${PATCH_DIR}/czi_100_datalist.json";                REPEAT_TRAIN=1 ;;
  50)  DATALIST="${PATCH_DIR}/czi_50percent/czi_100_datalist.json";  REPEAT_TRAIN=2 ;;
  10)  DATALIST="${PATCH_DIR}/czi_10percent/czi_100_datalist.json";  REPEAT_TRAIN=9 ;;
  *) echo "Unknown PERCENT: $PERCENT"; exit 1 ;;
esac
OUT="/cluster/projects/pmccaiteam/<USERNAME>/projects/cryoet/experiments/baselines_out/deepetpicker_czi_${PERCENT}pct"
export DATASET=czi DATALIST OUT REPEAT_TRAIN

echo "Datalist: $DATALIST"
echo "Output:   $OUT"
echo "Git: $(git -C "$REPO" rev-parse --short HEAD) on $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"

t=$(sbatch --parsable --time=3:00:00 --output="$LOGS/%x-%j.out" --export=ALL \
      "$KIT/submit_h4h_bench_train.sbatch")
c=$(sbatch --parsable --dependency=afterok:$t --array=0-19%4 --output="$LOGS/%x-%A_%a.out" --export=ALL \
      "$KIT/submit_h4h_bench_ckpt.sbatch")
s=$(sbatch --parsable --dependency=afterok:$c --output="$LOGS/%x-%j.out" --export=ALL \
      "$KIT/submit_h4h_bench_select.sbatch")
echo "czi ${PERCENT}%: train $t -> ckpt array $c -> select $s"
echo "result: $OUT/predictions/<best checkpoint>/metrics_czi.json (test at val-chosen thresholds)"
