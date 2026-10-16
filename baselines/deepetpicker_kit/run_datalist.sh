#!/bin/bash
# End-to-end from a detection datalist JSON:
#   prepare -> preprocess -> train config -> train -> infer -> score.
#
# Usage:  [EPOCHS=60 VENV=... DEP=...] bash run_datalist.sh <datalist.json> <out_dir>
#
# <out_dir> becomes DeepETPicker's base_path and holds everything it generates:
#   raw_data/ coords/ data_std/ gaussian<N>/ data_ocp/ runs/
set -euo pipefail
DATALIST="${1:?usage: bash run_datalist.sh <datalist.json> <out_dir>}"
OUT="${2:?usage: bash run_datalist.sh <datalist.json> <out_dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${VENV:=/scratch/cyyu/venvs/deepetpicker}"
: "${DEP:=/scratch/cyyu/cryodino/DeepETPicker}"
: "${DSET:=Dataset440}"
: "${EPOCHS:=60}"
: "${BATCH:=8}"
: "${BLOCK:=72}"
: "${PAD:=12}"
: "${LR:=0.001}"
: "${THRESH:=0.5}"
: "${GPU:=0}"
: "${LABEL_DIAMETER:=9}"

[ -f "$VENV/bin/activate" ] || { echo "ERROR: venv not found at $VENV -- run setup.sh first"; exit 1; }
[ -d "$DEP/bin" ]           || { echo "ERROR: DeepETPicker not found at $DEP -- run setup.sh first"; exit 1; }
module load python/3.11 2>/dev/null || true
source "$VENV/bin/activate"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
nvidia-smi -L 2>/dev/null || echo "(no GPU visible -- training/inference need one)"

echo "=== 1) datalist -> whole tomograms + coords ==="
python "$HERE/prepare_from_datalist.py" --datalist "$DATALIST" --out "$OUT"

echo "=== 2) preprocessing config ==="
CFGVARS="$(python "$HERE/make_configs.py" --out "$OUT" --dset-name "${DSET}_preprocess" \
              --label-diameter "$LABEL_DIAMETER" | tee /dev/stderr | grep -E '^(TRAIN_SET_IDS|VAL_SET_IDS|TEST_TOMO|VAL_TOMO|LABEL_NAME)=')"
eval "$CFGVARS"
PRE_CONFIG="$OUT/${DSET}_preprocess.py"

echo "=== 3) preprocess (coords, normalisation, labels, occupancy) ==="
python -u "$DEP/bin/preprocess.py" --pre_configs "$PRE_CONFIG"

echo "=== 4) training config (train ids $TRAIN_SET_IDS, val id $VAL_SET_IDS) ==="
python -u "$DEP/bin/generate_train_config.py" \
    --pre_configs "$PRE_CONFIG" \
    --dset_name "${DSET}_train" \
    --cfg_save_path "$OUT" \
    --train_set_ids "$TRAIN_SET_IDS" \
    --val_set_ids "$VAL_SET_IDS" \
    --batch_size "$BATCH" \
    --block_size "$BLOCK" \
    --pad_size "$PAD" \
    --learning_rate "$LR" \
    --max_epoch "$EPOCHS" \
    --threshold "$THRESH" \
    --gpu_id $GPU
TRAIN_CONFIG="$OUT/${DSET}_train.py"

# test.py and utils/misc.py write a `result/` tree relative to the cwd -- keep it in $OUT.
cd "$OUT"

echo "=== 5) train ($EPOCHS epochs) ==="
# Marker so step 6 can tell this run's checkpoint from one left by an earlier run in the same
# $OUT -- otherwise a failed training would silently be scored using a stale checkpoint.
MARKER="$(mktemp "$OUT/.train_start.XXXXXX")"
python -u "$DEP/bin/train_bash.py" --train_configs "$TRAIN_CONFIG"

echo "=== 6) pick checkpoint ==="
# ModelCheckpoint(save_top_k=1, monitor='val_loss') keeps exactly one file per run, named by
# PL's default template (no metric in the name), so take the newest .ckpt -- but only one this
# run actually wrote.
CKPT="$(find "$OUT/runs/${DSET}_train" -name '*.ckpt' -newer "$MARKER" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)"
rm -f "$MARKER"
if [ -z "$CKPT" ]; then
  echo "ERROR: training wrote no checkpoint under $OUT/runs/${DSET}_train."
  echo "       (Older checkpoints from previous runs are ignored on purpose -- scoring one of"
  echo "        those would report metrics for a model this run did not train.)"
  exit 1
fi
echo "checkpoint: $CKPT"

echo "=== 7) inference on every tomogram ==="
python -u "$DEP/bin/test_bash.py" --train_configs "$TRAIN_CONFIG" --checkpoints "$CKPT"
PRED_DIR="$(dirname "$(dirname "$CKPT")")/PredictedLabels"
[ -d "$PRED_DIR" ] || { echo "ERROR: no predictions at $PRED_DIR"; exit 1; }

echo "=== 8) score ($TEST_TOMO = test, $VAL_TOMO = val) ==="
python -u "$HERE/evaluate.py" --base "$OUT" --pred-dir "$PRED_DIR"

echo "=== DONE."
echo "  predictions: $PRED_DIR/Coords_withArea/"
echo "  metrics:     $PRED_DIR/metrics.json"
echo "  tensorboard: $OUT/runs/${DSET}_train/"
