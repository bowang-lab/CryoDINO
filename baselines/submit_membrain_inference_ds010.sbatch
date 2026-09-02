#!/bin/bash
# Run membrain_inference.py (the shared CryoDINO-schema baseline scorer) on the
# already-trained DS010 membrain checkpoint. Then: sbatch submit_membrain_inference_ds010.sbatch
#SBATCH --job-name=membrain_inference_ds010
#SBATCH -p gpu_pmcc_ai_team
#SBATCH --account=pmcc_ai_team_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/projects/bwanggroup/reza/projects/cryoet/slurm_logs/%x_%j.log
set -euo pipefail

date
hostname
nvidia-smi -L || true

BASELINES_DIR="/cluster/home/t139212uhn/scripts/cryoet/CryoDINO/baselines"
VENV="$BASELINES_DIR/membrain_kit/venv"
RUN_OUT="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/baselines_out/membrain_ds010"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cryodino
source "$VENV/bin/activate"

python "$BASELINES_DIR/membrain_inference.py" \
  --checkpoint "$RUN_OUT/checkpoints/membrain_binary-epoch=90-val_loss=0.710.ckpt" \
  --datalist   "$RUN_OUT/test_list.json" \
  --output-dir "$RUN_OUT/cryodino_metrics"

echo "Done: $RUN_OUT/cryodino_metrics"
date
