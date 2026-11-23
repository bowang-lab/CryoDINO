#!/bin/bash
# End-to-end membrain-seg baseline on Dataset001 (CZII-10001), from the existing MONAI datalist
# JSON. membrain-seg has no multi-class head, so it's scoped to membrane-only (raw label value 3)
# rather than lumping cytoplasm+organelles+membrane into one foreground mask -- see
# baselines/membrain_kit/prepare_from_datalist.py's --target-class.
# One-time setup (run on a login node with internet, see baselines/README notes) must be
# done first: build membrain_kit/venv. Then: sbatch submit_membrain_ds001.sbatch
#SBATCH --job-name=membrain_ds001
#SBATCH -p gpu_pmccaiteam
#SBATCH --account=pmccaiteam_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=220G
#SBATCH --time=30:00:00
##SBATCH --mail-user=<YOUR_EMAIL>
##SBATCH --mail-type=ALL
#SBATCH --output=slurm_logs/%x_%j.log
set -euo pipefail
: "${CRYOET_ROOT:?set CRYOET_ROOT to the CryoDINO data folder (holding datasets/ and experiments/), e.g. export CRYOET_ROOT=/path/to/cryoet}"

date
hostname
nvidia-smi -L || true

BASELINES_DIR="/cluster/home/$USER/scripts/cryoet/CryoDINO/baselines"
KIT_DIR="$BASELINES_DIR/membrain_kit"
VENV="$KIT_DIR/venv"
TARGET_CLASS=3   # Dataset001's membrane is raw label value 3

DATALIST="${CRYOET_ROOT}/experiments/Dataset001_CZII_10001_patches512_100_datalist.json"
OUT="${CRYOET_ROOT}/experiments/baselines_out/membrain_ds001"
mkdir -p "$OUT"

EPOCHS=100 ITERS=300 VENV="$VENV" TARGET_CLASS="$TARGET_CLASS" \
  bash "$KIT_DIR/run_datalist.sh" "$DATALIST" "$OUT"

echo "Done: $OUT"

echo "=== Scoring with CryoDINO-schema metrics (membrain_inference.py) ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cryodino
source "$VENV/bin/activate"

# Same best-checkpoint-by-lowest-val_loss selection as membrain_kit/run_datalist.sh, so this
# doesn't hardcode a filename that only matches one particular training run.
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

# test_list.json carries the ORIGINAL (multi-class) label paths verbatim -- --target-class
# extracts membrane-only from them at scoring time, matching what the model was trained on.
python "$BASELINES_DIR/membrain_inference.py" \
  --checkpoint "$CKPT" \
  --datalist   "$OUT/test_list.json" \
  --output-dir "$OUT/cryodino_metrics" \
  --target-class "$TARGET_CLASS" \
  --cpu-metrics

echo "Done: $OUT/cryodino_metrics"
date
