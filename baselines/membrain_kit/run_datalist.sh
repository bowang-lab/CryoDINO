#!/bin/bash
# End-to-end from a datalist JSON: prepare -> train -> infer on test -> save per-sample output.
#
# Usage:  [EPOCHS=100 ITERS=300 VENV=/path/venv] bash run_datalist.sh <datalist.json> <out_dir>
set -euo pipefail
DATALIST="${1:?usage: bash run_datalist.sh <datalist.json> <out_dir>}"
OUT="${2:?usage: bash run_datalist.sh <datalist.json> <out_dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${VENV:=$HERE/venv}"
: "${EPOCHS:=100}"
: "${ITERS:=300}"

[ -f "$VENV/bin/activate" ] || { echo "ERROR: venv not found at $VENV — build it (see README) or set VENV=."; exit 1; }
module load python/3.11 2>/dev/null || true
source "$VENV/bin/activate"
mkdir -p "$OUT"
nvidia-smi -L 2>/dev/null || echo "(no GPU visible — training/inference need one)"

echo "=== 1) prepare data_dir + test list from datalist ==="
python "$HERE/prepare_from_datalist.py" --datalist "$DATALIST" \
    --data-dir "$OUT/data_dir" --test-out "$OUT/test_list.json"

echo "=== 2) train ($EPOCHS epochs x $ITERS iters/epoch) ==="
python "$HERE/train_membrain.py" --data-dir "$OUT/data_dir" \
    --ckpt-dir "$OUT/checkpoints" --log-dir "$OUT/logs" \
    --max-epochs "$EPOCHS" --iters-per-epoch "$ITERS"

echo "=== 3) pick best checkpoint (lowest val_loss) ==="
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

echo "=== 4) infer on test + save per-sample output ==="
python "$HERE/infer_and_save.py" --test-list "$OUT/test_list.json" --ckpt "$CKPT" --out "$OUT"
echo "=== DONE. per-test outputs under $OUT/test_predictions/ ==="
