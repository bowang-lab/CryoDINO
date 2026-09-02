#!/bin/bash
# Run deepict_inference.py (the shared CryoDINO-schema baseline scorer) on the
# already-trained DS010 DeePiCt model. Then: sbatch submit_deepict_inference_ds010.sbatch
#SBATCH --job-name=deepict_inference_ds010
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
VENV="$BASELINES_DIR/deepict/venv"
DEEPICT_ROOT="$BASELINES_DIR/deepict/DeePiCt"
RUN_OUT="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/baselines_out/deepict_ds010"
DATALIST="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/Dataset010_CZII_10010_patches512_100_datalist.json"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cryodino
source "$VENV/bin/activate"

python "$BASELINES_DIR/deepict_inference.py" \
  --model-path   "$RUN_OUT/out/model_best.pth" \
  --train-config "$RUN_OUT/config.yaml" \
  --deepict-root "$DEEPICT_ROOT" \
  --datalist     "$DATALIST" \
  --output-dir   "$RUN_OUT/cryodino_metrics" \
  --threshold    0.5

echo "Done: $RUN_OUT/cryodino_metrics"
date
