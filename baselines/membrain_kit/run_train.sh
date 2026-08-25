#!/bin/bash
# Full membrain-seg training run (binary). Submit AFTER run_smoke.sh passes.
# ===== EDIT THESE =====
#SBATCH --job-name=membrain_train
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --gres=gpu:1                 # e.g. gpu:1 | gpu:h100:1 | gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
VENV=/path/to/venv
DATA_DIR=/path/to/Dataset_membrain
KIT_DIR=/path/to/membrain_kit
WORK_DIR=/path/to/workdir            # checkpoints/ and logs/ are written here (use fast scratch)
MAX_EPOCHS=100
ITERS_PER_EPOCH=300
# ======================

set -euo pipefail
module load python/3.11 2>/dev/null || true
source "$VENV/bin/activate"
cd "$WORK_DIR"; mkdir -p logs checkpoints
nvidia-smi -L

python -u "$KIT_DIR/train_membrain.py" \
  --data-dir "$DATA_DIR" \
  --max-epochs "$MAX_EPOCHS" \
  --iters-per-epoch "$ITERS_PER_EPOCH" \
  --batch-size 2 \
  --num-workers 10 \
  --project-name membrain --sub-name binary \
  --log-dir "$WORK_DIR/logs" --ckpt-dir "$WORK_DIR/checkpoints"

# Best checkpoints -> $WORK_DIR/checkpoints/membrain_binary-*.ckpt
# Copy them off scratch (often auto-purged) to durable storage when the run finishes.
