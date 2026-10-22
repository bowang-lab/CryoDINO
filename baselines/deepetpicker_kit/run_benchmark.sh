#!/bin/bash
# DeepETPicker on the CryoDINO detection benchmark split (train Dataset440 patches, val
# Dataset445, test Dataset446), scored like detection3d.py:
#   prepare -> preprocess -> train config -> train -> infer every val/test tomogram -> score.
# run_datalist.sh is the original single-split flow (val/test = 2 of the training tomograms).
#
# Usage:  [EPOCHS=60 VENV=... DEP=... REPO=...] bash run_benchmark.sh <datalist.json> <out_dir>
set -euo pipefail
DATALIST="${1:?usage: bash run_benchmark.sh <datalist.json> <out_dir>}"
OUT="${2:?usage: bash run_benchmark.sh <datalist.json> <out_dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${VENV:?set VENV}"
: "${DEP:?set DEP}"
: "${REPO:?set REPO (CryoDINO checkout, for metrics.py)}"
: "${DSET:=Dataset440}"
: "${EPOCHS:=60}"
: "${BATCH:=8}"
: "${BLOCK:=72}"
: "${PAD:=12}"
: "${LR:=0.001}"
: "${THRESH:=0.5}"
: "${GPU:=0}"
: "${LABEL_DIAMETER:=9}"
: "${WORKERS:=16}"
: "${ORIG_TRAIN_NII:=}"   # optional: dir of the original Dataset440 .nii.gz, for the axis check

source "$VENV/bin/activate"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
nvidia-smi -L 2>/dev/null || echo "(no GPU visible -- training/inference need one)"

echo "=== 1) benchmark split -> raw_data (train + 1 val), data_std (val/test), gt ==="
CFGVARS="$(python -u "$HERE/prepare_benchmark.py" --datalist "$DATALIST" --out "$OUT" \
              --dset-name "${DSET}_preprocess" --label-diameter "$LABEL_DIAMETER" \
              --workers "$WORKERS" ${ORIG_TRAIN_NII:+--check-orientation "$ORIG_TRAIN_NII"} \
           | tee /dev/stderr | grep -E '^(TRAIN_SET_IDS|VAL_SET_IDS)=')"
eval "$CFGVARS"
PRE_CONFIG="$OUT/${DSET}_preprocess.py"

echo "=== 2) preprocess the labelled tomograms (coords, normalisation, labels, occupancy) ==="
python -u "$DEP/bin/preprocess.py" --pre_configs "$PRE_CONFIG"

echo "=== 3) training config (train ids $TRAIN_SET_IDS, DeepETPicker val id $VAL_SET_IDS) ==="
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

cd "$OUT"   # test.py / utils/misc.py write a result/ tree relative to the cwd

echo "=== 4) train ($EPOCHS epochs) ==="
MARKER="$(mktemp "$OUT/.train_start.XXXXXX")"
python -u "$DEP/bin/train_bash.py" --train_configs "$TRAIN_CONFIG"

echo "=== 5) pick checkpoint ==="
CKPT="$(find "$OUT/runs/${DSET}_train" -name '*.ckpt' -newer "$MARKER" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)"
rm -f "$MARKER"
[ -n "$CKPT" ] || { echo "ERROR: training wrote no checkpoint under $OUT/runs/${DSET}_train"; exit 1; }
echo "checkpoint: $CKPT"

echo "=== 6) inference on every tomogram in data_std ($(ls "$OUT/data_std"/*.mrc | wc -l)) ==="
python -u "$DEP/bin/test_bash.py" --train_configs "$TRAIN_CONFIG" --checkpoints "$CKPT"
PRED_DIR="$(dirname "$(dirname "$CKPT")")/PredictedLabels"
[ -d "$PRED_DIR" ] || { echo "ERROR: no predictions at $PRED_DIR"; exit 1; }

echo "=== 7) score: thresholds on val, applied to test (CZIDetectionMetrics) ==="
python -u "$HERE/score_benchmark.py" --base "$OUT" --pred-dir "$PRED_DIR" --repo "$REPO"

echo "=== DONE."
echo "  predictions: $PRED_DIR/Coords_withArea/"
echo "  results:     $PRED_DIR/results.json"
