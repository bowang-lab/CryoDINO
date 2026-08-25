#!/bin/bash
# End-to-end from a datalist JSON: prepare (mrc+metadata+config) -> partition+train
# (EPOCHS x ITERS) -> infer on every test tomogram -> save one probability map per sample.
#
# Usage:  [EPOCHS=100 ITERS=300 DEEPICT=/path VENV=/path] bash run_datalist.sh <datalist.json> <out_dir>
# Prereq: setup.sh has been run (DeePiCt cloned + patched) and the venv is built (see README).
set -euo pipefail
DATALIST="${1:?usage: bash run_datalist.sh <datalist.json> <out_dir>}"
OUT="${2:?usage: bash run_datalist.sh <datalist.json> <out_dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DEEPICT:=$HERE/DeePiCt}"
: "${VENV:=$HERE/venv}"
: "${EPOCHS:=100}"
: "${ITERS:=300}"
export DEEPICT VENV

[ -d "$DEEPICT/3d_cnn" ] || { echo "ERROR: DeePiCt not found at $DEEPICT — run setup.sh."; exit 1; }
[ -f "$VENV/bin/activate" ] || { echo "ERROR: venv not found at $VENV — see README."; exit 1; }
module load python/3.11 2>/dev/null || true
source "$VENV/bin/activate"
mkdir -p "$OUT"

echo "=== 1) prepare mrc + metadata.csv + config.yaml from datalist ==="
python "$HERE/prepare_from_datalist.py" --datalist "$DATALIST" --out "$OUT" \
    --epochs "$EPOCHS" --iters-per-epoch "$ITERS"

echo "=== 2) partition training tomograms + train ($EPOCHS x $ITERS) ==="
bash "$HERE/run_pipeline.sh" "$OUT/config.yaml"

echo "=== 3) infer on test tomograms + save per-sample probability maps ==="
bash "$HERE/predict.sh" "$OUT/config.yaml"
echo "=== DONE. outputs under $OUT/out/predictions/ ==="
