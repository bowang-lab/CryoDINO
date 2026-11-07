#!/bin/bash
# membrain-seg baselines at 10% training data -- one job, all datasets that have a membrane class:
#   Dataset001  membrane = raw label value 3
#   Dataset010  already binary membrane (label > 0, same as submit_membrain_ds010.sbatch)
#   Dataset049  membranes = raw label value 2 (before the CryoDINO remap)
# EMPIAR-10989 is skipped: its only foreground is actin filaments, there is no membrane class.
#
# membrain-seg has no multi-class head, so each dataset is scoped to membranes only with
# --target-class (see baselines/membrain_kit/prepare_from_datalist.py).
# The "<base>_10percent_100_datalist.json" files carry the 10% training subset; their val/test are
# the full 100% sets, so scoring is on the same test tomograms as every other run.
#
# One-time setup (run on a login node with internet, see baselines/README notes) must be
# done first: build membrain_kit/venv. Then: sbatch submit_membrain_10pct.sbatch
#SBATCH --job-name=membrain_10pct
#SBATCH -p gpu_pmccaiteam
#SBATCH --account=pmccaiteam_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=220G
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/projects/pmccaiteam/reza/projects/cryoet/slurm_logs/%x_%j.log
# No `set -e`: a failure in one dataset is reported and the loop moves on to the next one.
set -uo pipefail

date
hostname
nvidia-smi -L || true

BASELINES_DIR="/cluster/home/t139212uhn/scripts/cryoet/CryoDINO/baselines"
KIT_DIR="$BASELINES_DIR/membrain_kit"
VENV="$KIT_DIR/venv"
EXP="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cryodino
source "$VENV/bin/activate"

DATASETS=(ds001 ds010 ds049)
declare -A BASE_NAME TARGET
BASE_NAME[ds001]="Dataset001_CZII_10001_patches512"
TARGET[ds001]=3     # membrane
BASE_NAME[ds010]="Dataset010_CZII_10010_patches512"
TARGET[ds010]=""    # already binary: label > 0
BASE_NAME[ds049]="Dataset049_EMPIAR_12049_transposed_patches512"
TARGET[ds049]=2     # membranes (raw value, pre-remap)

FAILED=()
for ds in "${DATASETS[@]}"; do
    DATALIST="$EXP/${BASE_NAME[$ds]}_10percent_100_datalist.json"
    OUT="$EXP/baselines_out/membrain_${ds}_10pct"
    TC="${TARGET[$ds]}"
    mkdir -p "$OUT"

    echo "=============================================="
    echo "membrain-seg 10% | $ds | target class: ${TC:-binary (label > 0)}"
    echo "Datalist: $DATALIST"
    echo "Output:   $OUT"
    echo "=============================================="

    if ! EPOCHS=100 ITERS=300 VENV="$VENV" TARGET_CLASS="$TC" \
         bash "$KIT_DIR/run_datalist.sh" "$DATALIST" "$OUT"; then
        echo "FAILED (training): $ds"; FAILED+=("$ds"); continue
    fi
    echo "Done training: $OUT"

    echo "=== Scoring with CryoDINO-schema metrics (membrain_inference.py): $ds ==="
    # Same best-checkpoint-by-lowest-val_loss selection as membrain_kit/run_datalist.sh.
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
    if [ -z "$CKPT" ]; then
        echo "FAILED (no checkpoint found in $OUT/checkpoints): $ds"; FAILED+=("$ds"); continue
    fi
    echo "best ckpt: $CKPT"

    # test_list.json carries the ORIGINAL label paths verbatim; --target-class extracts the membrane
    # class from them at scoring time, matching what the model was trained on.
    SCORE_ARGS=(--checkpoint "$CKPT" --datalist "$OUT/test_list.json" --output-dir "$OUT/cryodino_metrics" --cpu-metrics)
    if [ -n "$TC" ]; then
        SCORE_ARGS+=(--target-class "$TC")
    fi
    if ! python "$BASELINES_DIR/membrain_inference.py" "${SCORE_ARGS[@]}"; then
        echo "FAILED (scoring): $ds"; FAILED+=("$ds"); continue
    fi
    echo "Done: $OUT/cryodino_metrics"
done

echo "=============================================="
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Datasets that FAILED: ${FAILED[*]}"
    date
    exit 1
fi
echo "All datasets completed!"
date
