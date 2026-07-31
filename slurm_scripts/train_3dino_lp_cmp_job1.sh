#!/bin/bash
#SBATCH -J 3dino-lp-cmp-job1-h100pre
#SBATCH -p gpu_pmcc_ai_team
#SBATCH -t 7-00:00:00
#SBATCH --account=pmcc_ai_team_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

# Job 1: H100 pretraining checkpoints (all eval iterations) x 3 datasets.
# Thin wrapper — all logic lives in the shared engine script; we just pass labels.
bash /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/slurm_scripts/train_3dino_lp_h100_b200_comparison.sh \
    h100_pretrain
