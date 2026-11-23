#!/bin/bash
# Benchmark run on the datalist's real splits (manuscript protocol, EXPERIMENT_PLAN.md section 3),
# matched to CryoDINO's 3DINO/dinov2/eval/detection3d.py:
#   train      = every tomogram in the datalist `training` split (CZI 100%: all 7 of Dataset440)
#   budget     = 100 epochs x 300 iterations = 30k, batch 4, lr 5e-5 (DeepETPicker's CLI default),
#                a periodic checkpoint every 5 epochs (20 checkpoints)
#   ckpt sel.  = best score on the FULL `validation` split (submit_h4h_bench_ckpt + select_checkpoint.py)
#   reported   = test at the val-chosen thresholds; CZIDetectionMetrics F4 (czi) / BYUDetectionMetrics F2 (byu)
#
# The defaults below are the settings of the reported runs. The only per-run setting left is the
# training-set repeat for CZI subsets: REPEAT_TRAIN=2 (50%), 9 (10%). BYU uses REPEAT_TRAIN=auto.
#
# Usage:  [DATASET=czi|byu VENV=... DEP=...] bash run_benchmark.sh <datalist.json> <out_dir>
#
# STAGE=all (default) | train | eval
#   train: steps 1-6, writes <out_dir>/checkpoint_used.txt
#   eval:  steps 7-8 on CKPT, or on <out_dir>/checkpoint_used.txt when CKPT is unset.
#          Inference resumes per tomogram, so a timed-out eval can simply be resubmitted.
set -euo pipefail
DATALIST="${1:?usage: bash run_benchmark.sh <datalist.json> <out_dir>}"
OUT="${2:?usage: bash run_benchmark.sh <datalist.json> <out_dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${VENV:=$HOME/venvs/deepetpicker}"
: "${DEP:=$HOME/deepetpicker/DeepETPicker}"
: "${DATASET:=${EVAL_DATASET:-czi}}"   # czi | byu: data preparation (prepare_from_datalist / prepare_byu) + metric
case "$DATASET" in czi|byu) ;; *) echo "ERROR: DATASET must be czi|byu"; exit 1;; esac
EVAL_DATASET="$DATASET"                #   czi -> CZIDetectionMetrics (F4), byu -> BYUDetectionMetrics (F2)
: "${DSET:=$([ "$DATASET" = byu ] && echo BYU || echo Dataset440)}"
# --- training budget (same for both datasets, matched to CryoDINO) ---
: "${EPOCHS:=100}"
: "${ITERS_PER_EPOCH:=300}"    # cap each epoch at N batches (3DINO --epoch-length); 0 = full pass
: "${BATCH:=4}"
: "${LR:=0.00005}"             # 3e-4 and 1e-4 collapsed to one class at batch 4 (CZI); 3e-5 missed a class
: "${SAVE_EVERY:=5}"           # periodic checkpoint every N epochs (5 x 300 = 1,500 iterations -> 20)
# --- dataset-specific (czi = upstream DeepETPicker defaults; byu = the reported block-128 setup) ---
if [ "$DATASET" = byu ]; then
  : "${BLOCK:=128}"            # = CryoDINO's patch size; block 72 put false positives in empty background
  : "${LABEL_DIAMETER:=13}"    # beat 9 and 19 at 100%
  : "${CROP_MARGIN:=90}"       # training tomograms cropped to motors +- this; must be >= BLOCK/2 + BLOCK/5
  : "${INTERNAL_VAL:=all}"     # DeepETPicker's val_loss on every validation tomogram (prepare_byu.py)
  : "${DEP_FULL_VAL:=1}"       # ... which needs the patched train_bash.py (patch_periodic_ckpt.py)
  : "${REPEAT_TRAIN:=auto}"
else
  : "${BLOCK:=72}"
  : "${LABEL_DIAMETER:=9}"
  : "${CROP_MARGIN:=64}"       # unused for czi
  : "${INTERNAL_VAL:=0}"       # index or basename of the validation entry used for val_loss
  : "${DEP_FULL_VAL:=0}"
  : "${REPEAT_TRAIN:=1}"       # 50%: 2, 10%: 9 (the reported runs)
fi
#   REPEAT_TRAIN >1: an epoch shuffles over N copies of the training samples;
#                auto: smallest N with N*samples >= ITERS_PER_EPOCH*BATCH
: "${PAD:=12}"
: "${THRESH:=0.5}"
: "${GPU:=0}"
: "${SPLITS:=validation,test}"
: "${CHUNK:=16}"
: "${LIMIT:=}"              # e.g. LIMIT=4: only the first 4 tomograms per split (smoke)
: "${NII_ROOT:=}"
: "${PROFILE:=}"            # PROFILE=1: time NIfTI conversion + profile DeepETPicker inference
: "${CKPT:=}"
: "${BYU_MIN_RADIUS:=}"     # byu only: match radius in A (empty = metrics.py default, now 1000)
: "${STAGE:=all}"
case "$STAGE" in all|train|eval) ;; *) echo "ERROR: STAGE must be all|train|eval"; exit 1;; esac
[ "$STAGE" = train ] && [ -n "$CKPT" ] && { echo "ERROR: STAGE=train with CKPT set"; exit 1; }

[ -f "$VENV/bin/activate" ] || { echo "ERROR: venv not found at $VENV"; exit 1; }
[ -d "$DEP/bin" ]           || { echo "ERROR: DeepETPicker not found at $DEP"; exit 1; }
source "$VENV/bin/activate"
python -c "import nibabel" 2>/dev/null || { echo "ERROR: nibabel missing in $VENV -- pip install nibabel"; exit 1; }
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
TRAIN_CONFIG="$OUT/${DSET}_train.py"
LIMIT_ARG=(); [ -n "$LIMIT" ] && LIMIT_ARG=(--limit "$LIMIT")
PROF_ARG=();  [ -n "$PROFILE" ] && PROF_ARG=(--profile)
NII_ARG=();   [ -n "$NII_ROOT" ] && NII_ARG=(--nii-root "$NII_ROOT")
nvidia-smi -L 2>/dev/null || echo "(no GPU visible -- training/inference need one)"

if [ "$STAGE" = eval ] && [ -z "$CKPT" ]; then
  [ -f "$OUT/checkpoint_used.txt" ] || { echo "ERROR: STAGE=eval needs CKPT or $OUT/checkpoint_used.txt (run STAGE=train first)"; exit 1; }
  CKPT="$(cat "$OUT/checkpoint_used.txt")"
fi

if [ -z "$CKPT" ]; then
  echo "=== 1) datalist -> training tomograms + internal val tomogram ==="
  if [ "$DATASET" = byu ]; then
    # every voxel DeepETPicker's training can read lies within BLOCK/2 + BLOCK/5 of a motor
    [ "$CROP_MARGIN" -ge $((BLOCK / 2 + BLOCK / 5)) ] || { echo "ERROR: CROP_MARGIN=$CROP_MARGIN < BLOCK/2 + BLOCK/5 = $((BLOCK / 2 + BLOCK / 5))"; exit 1; }
    python "$HERE/prepare_byu.py" --datalist "$DATALIST" --out "$OUT" --crop-margin "$CROP_MARGIN" \
        --internal-val "$INTERNAL_VAL" ${NII_ARG[@]+"${NII_ARG[@]}"} --workers "${PREP_WORKERS:-3}"
  else
    python "$HERE/prepare_from_datalist.py" --datalist "$DATALIST" --out "$OUT" \
        --internal-val "$INTERNAL_VAL" ${NII_ARG[@]+"${NII_ARG[@]}"}
  fi

  echo "=== 2) preprocessing config ==="
  CFGVARS="$(python "$HERE/make_configs.py" --out "$OUT" --dset-name "${DSET}_preprocess" \
                --label-diameter "$LABEL_DIAMETER" | tee /dev/stderr | grep -E '^(TRAIN_SET_IDS|VAL_SET_IDS|VAL_TOMO)=')"
  eval "$CFGVARS"
  PRE_CONFIG="$OUT/${DSET}_preprocess.py"

  echo "=== 3) preprocess (coords, normalisation, labels, occupancy) ==="
  python -u "$DEP/bin/preprocess.py" --pre_configs "$PRE_CONFIG"
  if python -c "import json,sys; sys.exit(0 if json.load(open('$OUT/dataset_split.json')).get('prestandardized') else 1)"; then
    # crops were cut from whole-tomogram-standardised volumes (prepare_byu.py); preprocess.py just
    # re-standardised them with crop statistics -- put the whole-tomogram version back
    cp "$OUT"/raw_data/*.mrc "$OUT/data_std/"
    echo "data_std <- raw_data (whole-tomogram standardisation kept)"
  fi
  if [ "$REPEAT_TRAIN" = auto ]; then
    [ "$ITERS_PER_EPOCH" -gt 0 ] || { echo "ERROR: REPEAT_TRAIN=auto needs ITERS_PER_EPOCH>0"; exit 1; }
    REPEAT_TRAIN="$(python -c "
import json, math
s = json.load(open('$OUT/dataset_split.json'))
n = sum(s['particle_counts'][s['tomograms'][i]] for i in s['train_ids'])
print(max(1, math.ceil($ITERS_PER_EPOCH * $BATCH / n)))")"
    echo "REPEAT_TRAIN=auto -> $REPEAT_TRAIN"
  fi

  echo "=== 4) training config (train ids $TRAIN_SET_IDS, val id $VAL_SET_IDS = $VAL_TOMO) ==="
  python -u "$DEP/bin/generate_train_config.py" \
      --pre_configs "$PRE_CONFIG" --dset_name "${DSET}_train" --cfg_save_path "$OUT" \
      --train_set_ids "$TRAIN_SET_IDS" --val_set_ids "$VAL_SET_IDS" \
      --batch_size "$BATCH" --block_size "$BLOCK" --pad_size "$PAD" \
      --learning_rate "$LR" --max_epoch "$EPOCHS" --threshold "$THRESH" --gpu_id $GPU

  if [ "$SAVE_EVERY" -gt 0 ] || [ "$ITERS_PER_EPOCH" -gt 0 ] || [ "$REPEAT_TRAIN" -gt 1 ] || [ "$DEP_FULL_VAL" = 1 ]; then
    python "$HERE/patch_periodic_ckpt.py" --dep "$DEP"
  fi
  export DEP_SAVE_EVERY="$SAVE_EVERY" DEP_LIMIT_TRAIN_BATCHES="$ITERS_PER_EPOCH" DEP_REPEAT_TRAIN="$REPEAT_TRAIN" DEP_FULL_VAL
  cd "$OUT"   # train.py/test.py write a result/ tree relative to the cwd
  echo "=== 5) train ($EPOCHS epochs, lr $LR, batch $BATCH, iters/epoch ${ITERS_PER_EPOCH/#0/all}, training set x$REPEAT_TRAIN, checkpoint every ${SAVE_EVERY/#0/-} epochs, block $BLOCK, label $LABEL_DIAMETER, full val_loss ${DEP_FULL_VAL}) ==="
  MARKER="$(mktemp "$OUT/.train_start.XXXXXX")"
  python -u "$DEP/bin/train_bash.py" --train_configs "$TRAIN_CONFIG"

  echo "=== 6) pick checkpoint ==="
  # DeepETPicker's own best-val_loss checkpoint (not the periodic ones)
  CKPT="$(find "$OUT/runs/${DSET}_train" -name '*.ckpt' -not -path '*checkpoints_periodic*' -newer "$MARKER" \
          -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
  if [ "$SAVE_EVERY" -gt 0 ]; then
    find "$OUT/runs/${DSET}_train" -path '*checkpoints_periodic*' -name '*.ckpt' -newer "$MARKER" \
        | sort > "$OUT/periodic_checkpoints.txt"
    echo "periodic checkpoints: $(wc -l < "$OUT/periodic_checkpoints.txt") -> $OUT/periodic_checkpoints.txt"
    [ -s "$OUT/periodic_checkpoints.txt" ] || { echo "ERROR: SAVE_EVERY=$SAVE_EVERY but no periodic checkpoints were written"; exit 1; }
  fi
  rm -f "$MARKER"
  [ -n "$CKPT" ] || { echo "ERROR: training wrote no checkpoint under $OUT/runs/${DSET}_train"; exit 1; }
  echo "$CKPT" > "$OUT/checkpoint_used.txt"   # only training writes this (parallel eval jobs must not)
else
  echo "=== CKPT given: skipping steps 1-6 ==="
  [ -f "$CKPT" ] && [ -f "$TRAIN_CONFIG" ] || { echo "ERROR: need $CKPT and $TRAIN_CONFIG"; exit 1; }
fi
echo "checkpoint: $CKPT"
if [ "$STAGE" = train ]; then
  echo "=== STAGE=train done. checkpoint recorded in $OUT/checkpoint_used.txt"
  exit 0
fi

# one folder per checkpoint: version_N_<name>, same rule as infer_splits.py ckpt_tag()
PRED_ROOT="$OUT/predictions/$(basename "$(dirname "$(dirname "$CKPT")")")_$(basename "$CKPT" .ckpt)"
echo "=== 7) inference on the full $SPLITS splits ==="
python -u "$HERE/infer_splits.py" --datalist "$DATALIST" --out "$OUT" \
    --train-config "$TRAIN_CONFIG" --ckpt "$CKPT" --dep "$DEP" \
    --splits "$SPLITS" --chunk "$CHUNK" --pred-root "$PRED_ROOT" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"} ${NII_ARG[@]+"${NII_ARG[@]}"} ${PROF_ARG[@]+"${PROF_ARG[@]}"}

echo "=== 8) score ($EVAL_DATASET metric) ==="
python -u "$HERE/evaluate_czi.py" --datalist "$DATALIST" --pred-root "$PRED_ROOT" --dataset "$EVAL_DATASET" \
    --splits "$SPLITS" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"} ${BYU_MIN_RADIUS:+--byu-min-radius "$BYU_MIN_RADIUS"}

echo "=== DONE."
echo "  predictions: $PRED_ROOT/<split>/"
echo "  metrics:     $PRED_ROOT/metrics_${EVAL_DATASET}.json"
echo "  tensorboard: $OUT/runs/${DSET}_train/"
