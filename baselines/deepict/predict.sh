#!/bin/bash
# Run DeePiCt inference on every tomogram in the config's prediction_list, saving one probability
# map per (tomogram, semantic class). Requires a trained model (run_pipeline.sh first).
#
# Usage:  bash predict.sh <config.yaml>     (env: DEEPICT=, VENV=)
#
# Multi-class note: generate_prediction_partition.py and segment.py are CLASS-INDEPENDENT -- the
# network has one output channel per class and segment.py writes them all in one pass (its output
# path and done-marker carry only model+tomo, no class).  Only assemble_prediction.py is per-class:
# it extracts channel config.pred_class_number.  So we segment once per tomogram and assemble N
# times, and inference costs the same for N classes as for 1.
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
CLASSES=$(python -c "import yaml;print(' '.join(map(str,yaml.safe_load(open('$CONFIG'))['training']['semantic_classes'])))")
echo "classes: $CLASSES"

# assemble_prediction.py takes only --config_file, so give each class its own config whose
# prediction.semantic_class selects that channel.
CFGDIR="$(cd "$(dirname "$CONFIG")" && pwd)"
for c in $CLASSES; do
  python - "$CONFIG" "$c" "$CFGDIR/.predict_$c.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg["prediction"]["semantic_class"] = sys.argv[2]
yaml.safe_dump(cfg, open(sys.argv[3], "w"), sort_keys=False)
PY
done

for t in $TEST_TOMOS; do
  echo "=== predict $t ==="
  python "$SCRIPTS/generate_prediction_partition.py" --pythonpath "$SRC" --config_file "$CONFIG" --fold None --tomo_name "$t"
  # NO --gpu: keeps SLURM's (MIG) CUDA_VISIBLE_DEVICES intact (to_device(gpu=None))
  python "$SCRIPTS/segment.py"                       --pythonpath "$SRC" --config_file "$CONFIG" --fold None --tomo_name "$t"
  for c in $CLASSES; do
    echo "--- assemble $t / $c ---"
    python "$SCRIPTS/assemble_prediction.py"         --pythonpath "$SRC" --config_file "$CFGDIR/.predict_$c.yaml" --fold None --tomo_name "$t"
  done
done

MODEL=$(python -c "import os,yaml;print(os.path.basename(yaml.safe_load(open('$CONFIG'))['model_path'])[:-4])")
OUTDIR=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
echo "=== DONE. per-(test,class) probability maps: ==="
for c in $CLASSES; do
  ls "$OUTDIR/predictions/$MODEL"/*/"$c"/probability_map.mrc 2>/dev/null || true
done
