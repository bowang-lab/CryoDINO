#!/bin/bash
# End-to-end DeePiCt baseline on EMPIAR-10989 (actin filaments), 100% data, 200 epochs, from the
# existing MONAI datalist JSON. Binary foreground: BINARY=1 forces the single-class path even if the
# labels contain more than one foreground value, so this stays the same task as the other methods.
# One-time setup (run on a login node with internet, see baselines/README notes) must be
# done first: clone+patch DeePiCt (setup.sh) and build deepict/venv. Then:
# sbatch submit_deepict_ds989.sbatch
#SBATCH --job-name=deepict_ds989
#SBATCH -p gpu_pmccaiteam
#SBATCH --account=pmccaiteam_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --time=30:00:00
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/projects/pmccaiteam/reza/projects/cryoet/slurm_logs/%x_%j.log
set -euo pipefail

date
hostname
nvidia-smi -L || true

BASELINES_DIR="/cluster/home/t139212uhn/scripts/cryoet/CryoDINO/baselines"
KIT_DIR="$BASELINES_DIR/deepict"
DEEPICT="$KIT_DIR/DeePiCt"
VENV="$KIT_DIR/venv"

DATALIST="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/Dataset989_EMPIAR_10989_transposed_patches512_100_datalist.json"
OUT="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/baselines_out/deepict_ds989"
mkdir -p "$OUT"
cd "$OUT"

EPOCHS=200 ITERS=300 DEEPICT="$DEEPICT" VENV="$VENV" BINARY=1 bash "$KIT_DIR/run_datalist.sh" "$DATALIST" "$OUT"

echo "Done: $OUT"

echo "=== Scoring with CryoDINO-schema metrics (deepict_inference.py) ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cryodino
source "$VENV/bin/activate"

python "$BASELINES_DIR/deepict_inference.py" \
  --model-path   "$OUT/out/model_best.pth" \
  --train-config "$OUT/config.yaml" \
  --deepict-root "$DEEPICT" \
  --datalist     "$DATALIST" \
  --output-dir   "$OUT/cryodino_metrics" \
  --threshold    0.5 \
  --cpu-metrics

echo "Done: $OUT/cryodino_metrics"
date
