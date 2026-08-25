#!/bin/bash
# End-to-end membrain-seg baseline on DS010, from the existing MONAI datalist JSON.
# One-time setup (run on a login node with internet, see baselines/README notes) must be
# done first: build membrain_kit/venv. Then: sbatch submit_membrain_ds010.sbatch
#SBATCH --job-name=membrain_ds010
#SBATCH -p gpu_pmcc_ai_team
#SBATCH --account=pmcc_ai_team_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log
set -euo pipefail

date
hostname
nvidia-smi -L || true

BASELINES_DIR="/cluster/home/t139212uhn/scripts/cryoet/CryoDINO/baselines"
KIT_DIR="$BASELINES_DIR/membrain_kit"
VENV="$KIT_DIR/venv"

DATALIST="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/Dataset010_CZII_10010_patches512_100_datalist.json"
OUT="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/baselines_out/membrain_ds010"
mkdir -p "$OUT"

EPOCHS=100 ITERS=300 VENV="$VENV" bash "$KIT_DIR/run_datalist.sh" "$DATALIST" "$OUT"

echo "Done: $OUT"
date
