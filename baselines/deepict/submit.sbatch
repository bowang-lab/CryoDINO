#!/bin/bash
# Submit the DeePiCt pipeline (partition + train, EPOCHS x ITERS_PER_EPOCH) to SLURM.
# Edit the CAPITALS, then:  sbatch submit.sbatch
#
# RAM note: DeePiCt loads all partition boxes into memory (see README "Resources").
# 68 tomograms (~11.8k boxes) peaked at ~94 GB -> 192G is a safe default; scale with #boxes.
#SBATCH --job-name=deepict_train
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1   # or e.g. --gres=gpu:1 on non-MIG clusters
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
CONFIG="${CONFIG:-$HERE/myrun.yaml}"             # override: CONFIG=... sbatch submit.sbatch
export DEEPICT="${DEEPICT:-$HERE/DeePiCt}"
export VENV="${VENV:-$HERE/venv}"

module load python/3.11 2>/dev/null || true
nvidia-smi -L || true

# If your data isn't prepared yet, uncomment (dataset-specific; needs >=16 GB RAM):
# DATA="$HERE/data" bash "$HERE/prepare_data.sh"

bash "$HERE/run_pipeline.sh" "$CONFIG"
