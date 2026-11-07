#!/bin/bash
# End-to-end membrain-seg baseline on Dataset049 (EMPIAR-12049), from the existing MONAI datalist
# JSON. membrain-seg has no multi-class head, so it's scoped to membranes only (raw label value 2)
# rather than lumping membranes+microtubules+ribosomes (+matrix/centrioles) into one foreground
# mask -- see baselines/membrain_kit/prepare_from_datalist.py's --target-class. Raw labels are
# 0=bg 1=matrix 2=membranes 3=microtubules 4=ribosomes 5=centrioles; the CryoDINO remap
# (2->1, 3->2, 4->3, rest->bg) is not applied here, so the raw value 2 is the right one.
# One-time setup (run on a login node with internet, see baselines/README notes) must be
# done first: build membrain_kit/venv. Then: sbatch submit_membrain_ds049.sbatch
#SBATCH --job-name=membrain_ds049
#SBATCH -p gpu_pmccaiteam
#SBATCH --account=pmccaiteam_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=220G
#SBATCH --time=30:00:00
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/projects/pmccaiteam/reza/projects/cryoet/slurm_logs/%x_%j.log
set -euo pipefail

date
hostname
nvidia-smi -L || true

BASELINES_DIR="/cluster/home/t139212uhn/scripts/cryoet/CryoDINO/baselines"
KIT_DIR="$BASELINES_DIR/membrain_kit"
VENV="$KIT_DIR/venv"
TARGET_CLASS=2   # Dataset049's membranes are raw label value 2 (pre-remap)

DATALIST="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/Dataset049_EMPIAR_12049_transposed_patches512_100_datalist.json"
OUT="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/baselines_out/membrain_ds049"
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
# extracts membranes-only from them at scoring time, matching what the model was trained on.
python "$BASELINES_DIR/membrain_inference.py" \
  --checkpoint "$CKPT" \
  --datalist   "$OUT/test_list.json" \
  --output-dir "$OUT/cryodino_metrics" \
  --target-class "$TARGET_CLASS" \
  --cpu-metrics

echo "Done: $OUT/cryodino_metrics"
date
