#!/bin/bash
# End-to-end DeePiCt baseline on Dataset049 (EMPIAR-12049), 100% data, 200 epochs, from the existing
# MONAI datalist JSON. Multi-class with 3 classes (membranes, microtubules, ribosomes): the raw labels
# are 0=bg 1=matrix 2=membranes 3=microtubules 4=ribosomes 5=centrioles, and REMAP_12049=1 applies the
# CryoDINO remap (2->1, 3->2, 4->3, matrix/centrioles -> bg) before class detection, so the class
# order below is the remapped label values 1, 2, 3. deepict_inference.py gets --remap-12049 so the
# ground truth is remapped the same way when scoring.
# One-time setup (run on a login node with internet, see baselines/README notes) must be
# done first: clone+patch DeePiCt (setup.sh) and build deepict/venv. Then:
# sbatch submit_deepict_ds049.sbatch
#SBATCH --job-name=deepict_ds049
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

DATALIST="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/Dataset049_EMPIAR_12049_transposed_patches512_100_datalist.json"
OUT="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/baselines_out/deepict_ds049"
mkdir -p "$OUT"
cd "$OUT"

EPOCHS=200 ITERS=300 DEEPICT="$DEEPICT" VENV="$VENV" \
  REMAP_12049=1 CLASSES=membranes,microtubules,ribosomes \
  bash "$KIT_DIR/run_datalist.sh" "$DATALIST" "$OUT"

echo "Done: $OUT"

echo "=== Scoring with CryoDINO-schema metrics (deepict_inference.py) ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cryodino
source "$VENV/bin/activate"

# --train-config inherits semantic_classes (the 3 classes) from $OUT/config.yaml, which switches
# deepict_inference.py into multi-class scoring automatically.
python "$BASELINES_DIR/deepict_inference.py" \
  --model-path   "$OUT/out/model_best.pth" \
  --train-config "$OUT/config.yaml" \
  --deepict-root "$DEEPICT" \
  --datalist     "$DATALIST" \
  --output-dir   "$OUT/cryodino_metrics" \
  --remap-12049 \
  --cpu-metrics

echo "Done: $OUT/cryodino_metrics"
date
