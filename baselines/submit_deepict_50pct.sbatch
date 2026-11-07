#!/bin/bash
# DeePiCt baselines at 50% training data, 200 epochs -- one job, all four datasets:
#   Dataset001  multi-class: cytoplasm, organelles, membrane (raw labels 1, 2, 3)
#   Dataset010  binary (BINARY=1, any foreground value > 0)
#   EMPIAR-10989 binary (BINARY=1)
#   Dataset049  multi-class: membranes, microtubules, ribosomes, with REMAP_12049=1 (matrix/centrioles
#               -> bg, 2->1, 3->2, 4->3) and --remap-12049 when scoring
# The "<base>_50percent_100_datalist.json" files carry the 50% training subset; scoring uses the full
# 100% datalist, whose test split is the same for every percentage.
# One-time setup (run on a login node with internet, see baselines/README notes) must be
# done first: clone+patch DeePiCt (setup.sh) and build deepict/venv. Then:
# sbatch submit_deepict_50pct.sbatch
#SBATCH --job-name=deepict_50pct
#SBATCH -p gpu_pmccaiteam
#SBATCH --account=pmccaiteam_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/projects/pmccaiteam/reza/projects/cryoet/slurm_logs/%x_%j.log
# No `set -e`: a failure in one dataset is reported and the loop moves on to the next one.
set -uo pipefail

date
hostname
nvidia-smi -L || true

BASELINES_DIR="/cluster/home/t139212uhn/scripts/cryoet/CryoDINO/baselines"
KIT_DIR="$BASELINES_DIR/deepict"
DEEPICT="$KIT_DIR/DeePiCt"
VENV="$KIT_DIR/venv"
EXP="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cryodino
source "$VENV/bin/activate"

DATASETS=(ds001 ds010 ds989 ds049)
declare -A BASE_NAME CLS BIN REMAP
BASE_NAME[ds001]="Dataset001_CZII_10001_patches512"
CLS[ds001]="cytoplasm,organelles,membrane"; BIN[ds001]=""; REMAP[ds001]=""
BASE_NAME[ds010]="Dataset010_CZII_10010_patches512"
CLS[ds010]="";                              BIN[ds010]=1;  REMAP[ds010]=""
BASE_NAME[ds989]="Dataset989_EMPIAR_10989_transposed_patches512"
CLS[ds989]="";                              BIN[ds989]=1;  REMAP[ds989]=""
BASE_NAME[ds049]="Dataset049_EMPIAR_12049_transposed_patches512"
CLS[ds049]="membranes,microtubules,ribosomes"; BIN[ds049]=""; REMAP[ds049]=1

FAILED=()
for ds in "${DATASETS[@]}"; do
    DATALIST="$EXP/${BASE_NAME[$ds]}_50percent_100_datalist.json"
    FULL_DATALIST="$EXP/${BASE_NAME[$ds]}_100_datalist.json"
    OUT="$EXP/baselines_out/deepict_${ds}_50pct"
    mkdir -p "$OUT"
    # DeePiCt writes its .done_patterns/ markers relative to the working directory; a directory per run
    # keeps one run's markers from being picked up by another with the same tomogram names.
    cd "$OUT"

    echo "=============================================="
    echo "DeePiCt 50% | $ds | classes: ${CLS[$ds]:-binary} | remap12049: ${REMAP[$ds]:-no}"
    echo "Datalist: $DATALIST"
    echo "Output:   $OUT"
    echo "=============================================="

    if ! EPOCHS=200 ITERS=300 DEEPICT="$DEEPICT" VENV="$VENV" \
         CLASSES="${CLS[$ds]}" BINARY="${BIN[$ds]}" REMAP_12049="${REMAP[$ds]}" \
         bash "$KIT_DIR/run_datalist.sh" "$DATALIST" "$OUT"; then
        echo "FAILED (training/prediction): $ds"; FAILED+=("$ds"); continue
    fi
    echo "Done training: $OUT"

    echo "=== Scoring with CryoDINO-schema metrics (deepict_inference.py): $ds ==="
    SCORE_ARGS=(--model-path "$OUT/out/model_best.pth" --train-config "$OUT/config.yaml"
                --deepict-root "$DEEPICT" --datalist "$FULL_DATALIST"
                --output-dir "$OUT/cryodino_metrics" --cpu-metrics)
    if [ -n "${REMAP[$ds]}" ]; then
        SCORE_ARGS+=(--remap-12049)
    fi
    if [ -n "${BIN[$ds]}" ]; then
        SCORE_ARGS+=(--threshold 0.5)
    fi
    if ! python "$BASELINES_DIR/deepict_inference.py" "${SCORE_ARGS[@]}"; then
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
