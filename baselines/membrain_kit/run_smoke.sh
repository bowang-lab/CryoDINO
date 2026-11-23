#!/bin/bash
# Short GPU sanity job: assert membrain train_loss is FINITE before a long run.
# ===== EDIT THESE =====
#SBATCH --job-name=membrain_smoke
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --gres=gpu:1                 # e.g. gpu:1 | gpu:h100:1 | gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
VENV=/path/to/venv                   # your membrain-seg virtualenv
DATA_DIR=/path/to/Dataset_membrain   # imagesTr/labelsTr/imagesVal/labelsVal
KIT_DIR=/path/to/membrain_kit        # folder containing train_membrain.py
# ======================

set -euo pipefail
module load python/3.11 2>/dev/null || true   # adjust to your cluster's module system
source "$VENV/bin/activate"
nvidia-smi -L
python -u "$KIT_DIR/train_membrain.py" --data-dir "$DATA_DIR" --smoke
