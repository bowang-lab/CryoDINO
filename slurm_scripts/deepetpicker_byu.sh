#!/bin/bash
# =========================
# DeepETPicker baseline on BYU (flagellar motor), same protocol as detection3d.py:
#   train  every `training` tomogram of the datalist with a motor (whole tomograms, cropped around
#          the motors; prepare_byu.py), 100 epochs x 300 iterations = 30k, batch 4, lr 5e-5,
#          block 128, label 13, a checkpoint every 5 epochs
#   val    54 whole tomograms: scores the checkpoints, picks the best one and the threshold
#   test   55 whole tomograms: scored once with that checkpoint at that threshold
#   metric BYUDetectionMetrics (F2) from 3DINO/dinov2/eval/detection_3d/metrics.py
#
# Run on the LOGIN node (it only submits; three dependent jobs: train -> ckpt array -> select):
#   PERCENT=100 bash slurm_scripts/deepetpicker_byu.sh      (100 | 50 | 10)
# Training defaults live in baselines/deepetpicker_kit/run_benchmark.sh (DATASET=byu block).
#
# The first periodic checkpoint (epoch 5, file epoch=4) is skipped: barely trained, it predicts so
# many motors that its validation scoring does not finish. Array 1-19 + SKIP_CKPT_EPOCHS=4 leave it
# out of the selection, as in the reported runs.
#
# Env: same as deepetpicker_czi.sh.
# =========================
set -euo pipefail
: "${CRYOET_ROOT:?set CRYOET_ROOT to the CryoDINO data folder (holding datasets/ and experiments/), e.g. export CRYOET_ROOT=/path/to/cryoet}"
: "${PERCENT:?set PERCENT (100 | 50 | 10)}"
[ -z "${SLURM_JOB_ID:-}" ] || { echo "ERROR: run this with bash on the login node, not sbatch (it submits the jobs itself)"; exit 1; }

REPO="${REPO:-/cluster/home/$USER/projects/CryoDINO}"
export KIT="$REPO/baselines/deepetpicker_kit"
export VENV="${VENV:-${CRYOET_ROOT}/envs/deepetpicker}"
export DEP="${DEP:-${CRYOET_ROOT}/envs/DeepETPicker}"
LOGS="${LOGS:-$PWD/slurm_logs}"   # run from the repo root, like the #SBATCH --output=slurm_logs/ lines

[ -f "$KIT/run_benchmark.sh" ] || { echo "ERROR: no run_benchmark.sh under $KIT"; exit 1; }
[ -f "$VENV/bin/activate" ] || { echo "ERROR: DeepETPicker venv not built at $VENV"; exit 1; }
[ -d "$DEP" ] || { echo "ERROR: DeepETPicker checkout missing at $DEP"; exit 1; }
mkdir -p "$LOGS"

PATCH_DIR="${CRYOET_ROOT}/datasets/downstream_detection/byu_detection_patches_128_motoronly"
case "$PERCENT" in
  100)   DATALIST="${PATCH_DIR}/byu_100_datalist.json" ;;
  50|10) DATALIST="${PATCH_DIR}/byu_${PERCENT}percent/byu_100_datalist.json" ;;
  *) echo "Unknown PERCENT: $PERCENT"; exit 1 ;;
esac
OUT="${CRYOET_ROOT}/experiments/baselines_out/deepetpicker_byu_${PERCENT}pct"
export DATASET=byu DATALIST OUT

echo "Datalist: $DATALIST"
echo "Output:   $OUT"
echo "Git: $(git -C "$REPO" rev-parse --short HEAD) on $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"

t=$(sbatch --parsable --time=8:00:00 --output="$LOGS/%x-%j.out" --export=ALL \
      "$KIT/submit_h4h_bench_train.sbatch")
c=$(sbatch --parsable --dependency=afterok:$t --array=1-19%4 --output="$LOGS/%x-%A_%a.out" --export=ALL \
      "$KIT/submit_h4h_bench_ckpt.sbatch")
s=$(sbatch --parsable --time=3:00:00 --dependency=afterok:$c --output="$LOGS/%x-%j.out" \
      --export=ALL,CHUNK=2,SKIP_CKPT_EPOCHS=4 "$KIT/submit_h4h_bench_select.sbatch")
echo "byu ${PERCENT}%: train $t -> ckpt array $c -> select $s"
echo "result: $OUT/predictions/<best checkpoint>/metrics_byu.json (test at the val-chosen threshold)"
