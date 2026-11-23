#!/bin/bash
# DeePiCt baseline on one dataset: train with deepict/run_datalist.sh (200 epochs x 300 iterations),
# then score the test split with CryoDINO-schema metrics (deepict_inference.py).
#
#   sbatch --export=ALL,DATASET=<001|010|989|049>,PERCENT=<100|50|10> submit_deepict.sbatch
#   all of them:  for d in 001 010 989 049; do for p in 100 50 10; do
#                   sbatch --export=ALL,DATASET=$d,PERCENT=$p submit_deepict.sbatch; done; done
#
# Datasets:
#   001  multi-class: cytoplasm, organelles, membrane (raw labels 1, 2, 3)
#   010  binary (BINARY=1, any foreground value > 0)
#   989  binary (BINARY=1; actin filaments)
#   049  multi-class: membranes, microtubules, ribosomes, with REMAP_12049=1 (matrix/centrioles -> bg,
#        2->1, 3->2, 4->3) and --remap-12049 when scoring
# 50%/10% train on "<base>_<p>percent_100_datalist.json"; scoring always uses the full 100% datalist,
# whose test split is the same for every percentage.
#
# One-time setup (login node with internet, see deepict/README.md): clone+patch DeePiCt (setup.sh)
# and build deepict/venv.
#SBATCH --job-name=deepict
#SBATCH -p gpu_pmccaiteam
#SBATCH --account=pmccaiteam_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --time=30:00:00
#SBATCH --output=/cluster/projects/pmccaiteam/reza/projects/cryoet/slurm_logs/%x_%j.log
set -euo pipefail
: "${DATASET:?set DATASET (001 | 010 | 989 | 049)}"
: "${PERCENT:=100}"

BASELINES_DIR="${BASELINES_DIR:-/cluster/home/$USER/scripts/cryoet/CryoDINO/baselines}"
KIT_DIR="$BASELINES_DIR/deepict"
DEEPICT="$KIT_DIR/DeePiCt"
VENV="$KIT_DIR/venv"
EXP="${EXP:-/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments}"

CLASSES=""; BINARY=""; REMAP_12049=""
case "$DATASET" in
  001) BASE="Dataset001_CZII_10001_patches512";              CLASSES="cytoplasm,organelles,membrane" ;;
  010) BASE="Dataset010_CZII_10010_patches512";              BINARY=1 ;;
  989) BASE="Dataset989_EMPIAR_10989_transposed_patches512"; BINARY=1 ;;
  049) BASE="Dataset049_EMPIAR_12049_transposed_patches512"; CLASSES="membranes,microtubules,ribosomes"; REMAP_12049=1 ;;
  *) echo "Unknown DATASET: $DATASET"; exit 1 ;;
esac
FULL_DATALIST="$EXP/${BASE}_100_datalist.json"
case "$PERCENT" in
  100) DATALIST="$FULL_DATALIST"
       OUT="$EXP/baselines_out/deepict_ds${DATASET}"
       [ "$DATASET" = 010 ] && OUT="${OUT}_200ep" ;;   # the reported 010 run's folder
  50|10) DATALIST="$EXP/${BASE}_${PERCENT}percent_100_datalist.json"
         OUT="$EXP/baselines_out/deepict_ds${DATASET}_${PERCENT}pct" ;;
  *) echo "Unknown PERCENT: $PERCENT"; exit 1 ;;
esac

date
hostname
nvidia-smi -L || true
echo "DeePiCt $PERCENT% | ds$DATASET | classes: ${CLASSES:-binary} | remap12049: ${REMAP_12049:-no}"
echo "Datalist: $DATALIST"
echo "Output:   $OUT"

mkdir -p "$OUT"
# DeePiCt writes its .done_patterns/ markers relative to the working directory, so run from $OUT
cd "$OUT"
EPOCHS=200 ITERS=300 DEEPICT="$DEEPICT" VENV="$VENV" \
  CLASSES="$CLASSES" BINARY="$BINARY" REMAP_12049="$REMAP_12049" \
  bash "$KIT_DIR/run_datalist.sh" "$DATALIST" "$OUT"
echo "Done training: $OUT"

echo "=== Scoring with CryoDINO-schema metrics (deepict_inference.py) ==="
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate cryodino; source "$VENV/bin/activate"; set -u
# --train-config carries semantic_classes from $OUT/config.yaml, which selects multi-class scoring
SCORE_ARGS=(--model-path "$OUT/out/model_best.pth" --train-config "$OUT/config.yaml"
            --deepict-root "$DEEPICT" --datalist "$FULL_DATALIST"
            --output-dir "$OUT/cryodino_metrics" --cpu-metrics)
[ -n "$REMAP_12049" ] && SCORE_ARGS+=(--remap-12049)
[ -n "$BINARY" ] && SCORE_ARGS+=(--threshold 0.5)
python "$BASELINES_DIR/deepict_inference.py" "${SCORE_ARGS[@]}"
echo "Done: $OUT/cryodino_metrics"
date
