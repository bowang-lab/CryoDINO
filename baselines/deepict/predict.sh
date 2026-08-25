#!/bin/bash
# Run DeePiCt inference on every tomogram in the config's prediction_list, saving one
# probability map per sample. Requires a trained model (run_pipeline.sh first).
#
# Usage:  bash predict.sh <config.yaml>     (env: DEEPICT=, VENV=)
set -euo pipefail
CONFIG="${1:?usage: bash predict.sh <config.yaml>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DEEPICT:=$HERE/DeePiCt}"
: "${VENV:=$HERE/venv}"
SRC="$DEEPICT/3d_cnn/src"
SCRIPTS="$DEEPICT/3d_cnn/scripts"

[ -f "$VENV/bin/activate" ] || { echo "ERROR: venv not found at $VENV — see README, or set VENV=."; exit 1; }
module load python/3.11 2>/dev/null || true
source "$VENV/bin/activate"
export PYTHONPATH="$SRC"

TEST_TOMOS=$(python -c "import yaml;print(' '.join(map(str,yaml.safe_load(open('$CONFIG'))['tomos_sets']['prediction_list'])))")
[ -n "$TEST_TOMOS" ] || { echo "prediction_list is empty in $CONFIG"; exit 0; }

for t in $TEST_TOMOS; do
  echo "=== predict $t ==="
  python "$SCRIPTS/generate_prediction_partition.py" --pythonpath "$SRC" --config_file "$CONFIG" --fold None --tomo_name "$t"
  # NO --gpu: keeps SLURM's (MIG) CUDA_VISIBLE_DEVICES intact (to_device(gpu=None))
  python "$SCRIPTS/segment.py"                       --pythonpath "$SRC" --config_file "$CONFIG" --fold None --tomo_name "$t"
  python "$SCRIPTS/assemble_prediction.py"           --pythonpath "$SRC" --config_file "$CONFIG" --fold None --tomo_name "$t"
done

MODEL=$(python -c "import os,yaml;print(os.path.basename(yaml.safe_load(open('$CONFIG'))['model_path'])[:-4])")
OUTDIR=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
echo "=== DONE. per-test probability maps: ==="
echo "$OUTDIR/predictions/$MODEL/<tomo>/particle/probability_map.mrc"
ls "$OUTDIR/predictions/$MODEL"/*/particle/probability_map.mrc 2>/dev/null || true
