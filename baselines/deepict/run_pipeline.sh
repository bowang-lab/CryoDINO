#!/bin/bash
# Generic DeePiCt 3D-CNN pipeline: partition all training tomograms, then train
# (EPOCHS x ITERS_PER_EPOCH, from the config). Reusable across cryo-ET datasets.
#
# Usage:   bash run_pipeline.sh <config.yaml>
# Env override (optional): DEEPICT=<repo> VENV=<venv> bash run_pipeline.sh config.yaml
#
# Prerequisites: setup.sh has been run (repo cloned + patched), a venv exists, and the data is
# already in .mrc with a metadata.csv (see README: "Preparing a new dataset").
set -euo pipefail

CONFIG="${1:?usage: bash run_pipeline.sh <config.yaml>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DEEPICT:=$HERE/DeePiCt}"   # override with DEEPICT= if the repo lives elsewhere
: "${VENV:=$HERE/venv}"         # override with VENV= to point at your venv
SRC="$DEEPICT/3d_cnn/src"
SCRIPTS="$DEEPICT/3d_cnn/scripts"

[ -d "$DEEPICT/3d_cnn" ] || { echo "ERROR: DeePiCt not found at $DEEPICT — run setup.sh first (or set DEEPICT=)."; exit 1; }
[ -f "$VENV/bin/activate" ] || { echo "ERROR: venv not found at $VENV — build it (see README 'One-time setup', or set VENV=)."; exit 1; }
module load python/3.11 2>/dev/null || true
source "$VENV/bin/activate"
export PYTHONPATH="$SRC"

echo "=== GPU ==="; nvidia-smi -L || { echo "no GPU visible"; exit 1; }

echo "=== 1) partition each training tomogram -> work_dir/training_data/<t>/partition.h5 ==="
TRAIN_TOMOS=$(python -c "import yaml;print(' '.join(map(str,yaml.safe_load(open('$CONFIG'))['tomos_sets']['training_list'])))")
for t in $TRAIN_TOMOS; do
  echo "--- partition $t ---"
  python "$SCRIPTS/generate_training_data.py" --pythonpath "$SRC" \
      --config_file "$CONFIG" --fold None --tomo_name "$t"
done

echo "=== 2) train (fixed EPOCHS x ITERS_PER_EPOCH schedule) ==="
python "$HERE/deepict_train.py" --config_file "$CONFIG" --pythonpath "$SRC" --fold None

echo "=== DONE. model(s): ==="
python -c "import yaml;c=yaml.safe_load(open('$CONFIG'));print(c['model_path'])"
