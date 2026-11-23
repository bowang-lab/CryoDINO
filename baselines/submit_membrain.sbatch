#!/bin/bash
# MemBrain-seg baseline on one dataset: train with membrain_kit/run_datalist.sh (100 epochs x 300
# iterations), then score the test split with CryoDINO-schema metrics (membrain_inference.py).
#
#   sbatch --export=ALL,DATASET=<001|010|049>,PERCENT=<100|50|10> submit_membrain.sbatch
#   all of them:  for d in 001 010 049; do for p in 100 50 10; do
#                   sbatch --export=ALL,DATASET=$d,PERCENT=$p submit_membrain.sbatch; done; done
#
# MemBrain-seg has no multi-class head, so each dataset is scoped to membranes with --target-class
# (see membrain_kit/prepare_from_datalist.py):
#   001  membrane = raw label value 3
#   010  already binary membrane (label > 0)
#   049  membranes = raw label value 2 (before the CryoDINO remap)
# EMPIAR-10989 has no membrane class (actin only) and is not run.
# 50%/10% train on "<base>_<p>percent_100_datalist.json"; their val/test are the full 100% sets.
#
# One-time setup (login node with internet, see membrain_kit/README.md): build membrain_kit/venv.
#SBATCH --job-name=membrain
#SBATCH -p gpu_pmccaiteam
#SBATCH --account=pmccaiteam_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=220G
#SBATCH --time=30:00:00
#SBATCH --output=/cluster/projects/pmccaiteam/<USERNAME>/projects/cryoet/slurm_logs/%x_%j.log
set -euo pipefail
: "${DATASET:?set DATASET (001 | 010 | 049)}"
: "${PERCENT:=100}"

BASELINES_DIR="${BASELINES_DIR:-/cluster/home/$USER/scripts/cryoet/CryoDINO/baselines}"
KIT_DIR="$BASELINES_DIR/membrain_kit"
VENV="$KIT_DIR/venv"
EXP="${EXP:-/cluster/projects/pmccaiteam/<USERNAME>/projects/cryoet/experiments}"

case "$DATASET" in
  001) BASE="Dataset001_CZII_10001_patches512";              TARGET_CLASS=3 ;;
  010) BASE="Dataset010_CZII_10010_patches512";              TARGET_CLASS="" ;;
  049) BASE="Dataset049_EMPIAR_12049_transposed_patches512"; TARGET_CLASS=2 ;;
  989) echo "EMPIAR-10989 has no membrane class; MemBrain-seg is not run on it"; exit 1 ;;
  *) echo "Unknown DATASET: $DATASET"; exit 1 ;;
esac
case "$PERCENT" in
  100) DATALIST="$EXP/${BASE}_100_datalist.json"
       OUT="$EXP/baselines_out/membrain_ds${DATASET}"
       [ "$DATASET" = 010 ] && OUT="$EXP/baselines_out/membrain_ds010_100pct_rerun" ;;   # the reported 010 run's folder
  50|10) DATALIST="$EXP/${BASE}_${PERCENT}percent_100_datalist.json"
         OUT="$EXP/baselines_out/membrain_ds${DATASET}_${PERCENT}pct" ;;
  *) echo "Unknown PERCENT: $PERCENT"; exit 1 ;;
esac

date
hostname
nvidia-smi -L || true
echo "membrain-seg $PERCENT% | ds$DATASET | target class: ${TARGET_CLASS:-binary (label > 0)}"
echo "Datalist: $DATALIST"
echo "Output:   $OUT"

mkdir -p "$OUT"
EPOCHS=100 ITERS=300 VENV="$VENV" TARGET_CLASS="$TARGET_CLASS" \
  bash "$KIT_DIR/run_datalist.sh" "$DATALIST" "$OUT"
echo "Done training: $OUT"

echo "=== Scoring with CryoDINO-schema metrics (membrain_inference.py) ==="
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate cryodino; source "$VENV/bin/activate"; set -u
# Same best-checkpoint-by-lowest-val_loss selection as membrain_kit/run_datalist.sh.
CKPT=$(python - "$OUT/checkpoints" <<'PY'
import glob, os, re, sys
best, bv = None, 1e9
for f in glob.glob(os.path.join(sys.argv[1], "*.ckpt")):
    m = re.search(r"val_loss=([0-9]+\.[0-9]+)", os.path.basename(f))
    v = float(m.group(1)) if m else 1e9
    if v <= bv: best, bv = f, v
print(best or "")
PY
)
[ -n "$CKPT" ] || { echo "ERROR: no checkpoint found in $OUT/checkpoints"; exit 1; }
echo "best ckpt: $CKPT"

# test_list.json carries the original label paths; --target-class extracts the membrane class from
# them at scoring time, matching what the model was trained on.
SCORE_ARGS=(--checkpoint "$CKPT" --datalist "$OUT/test_list.json" --output-dir "$OUT/cryodino_metrics" --cpu-metrics)
[ -n "$TARGET_CLASS" ] && SCORE_ARGS+=(--target-class "$TARGET_CLASS")
python "$BASELINES_DIR/membrain_inference.py" "${SCORE_ARGS[@]}"
echo "Done: $OUT/cryodino_metrics"
date
