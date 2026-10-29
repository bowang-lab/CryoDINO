#!/bin/bash
# End-to-end DeePiCt baseline on Dataset001 (CZII-10001), from the existing MONAI datalist JSON.
# First real test of the merged multi-class kit (see baselines/deepict/README.md's "Binary and
# multi-class") -- 3 real classes (cytoplasm, organelles, membrane), no remap needed (DS001's
# raw labels are already 0=bg/1=cytoplasm/2=organelles/3=membrane, contiguous from 1).
# One-time setup (run on a login node with internet, see baselines/README notes) must be
# done first: clone+patch DeePiCt (setup.sh) and build deepict/venv. Then:
# sbatch submit_deepict_ds001.sbatch
#SBATCH --job-name=deepict_ds001
#SBATCH -p gpu_pmccaiteam
#SBATCH --account=pmccaiteam_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=220G
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

DATALIST="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/Dataset001_CZII_10001_patches512_100_datalist.json"
OUT="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/baselines_out/deepict_ds001"
mkdir -p "$OUT"

# Training already completed successfully (model_best.pth exists under $OUT/out/) -- only the
# scoring step below failed (missing --cpu-metrics, see next section). Commented out so this
# resubmission just reruns scoring against the existing trained model.
# EPOCHS=100 ITERS=300 DEEPICT="$DEEPICT" VENV="$VENV" \
#   CLASSES=cytoplasm,organelles,membrane \
#   bash "$KIT_DIR/run_datalist.sh" "$DATALIST" "$OUT"

echo "Done: $OUT"

echo "=== Scoring with CryoDINO-schema metrics (deepict_inference.py) ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cryodino
source "$VENV/bin/activate"

# --train-config inherits semantic_classes (the 3 real classes) from $OUT/config.yaml, which
# switches deepict_inference.py into multi-class scoring automatically -- no --classes flag here.
python "$BASELINES_DIR/deepict_inference.py" \
  --model-path   "$OUT/out/model_best.pth" \
  --train-config "$OUT/config.yaml" \
  --deepict-root "$DEEPICT" \
  --datalist     "$DATALIST" \
  --output-dir   "$OUT/cryodino_metrics" \
  --cpu-metrics

echo "Done: $OUT/cryodino_metrics"
date
