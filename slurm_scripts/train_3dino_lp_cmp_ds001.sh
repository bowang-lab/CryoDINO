#!/bin/bash
#SBATCH -J cryodino_lp-ds001
#SBATCH -p gpu_pmcc_ai_team
#SBATCH -t 7-00:00:00
#SBATCH --account=pmcc_ai_team_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

# One job per dataset: ALL backbones/checkpoints for Dataset001 only.
# This job exclusively owns the Dataset001 cache dir -> no cross-job cache race.
# num_workers halved (16->8): repeated OOMs on this dataset scaled with iteration
# count as memory increased (220G/440G/700G died progressively later, ~4800/8900/
# 9900 of 12500 iters) -> leak-like growth, not a single oversized item. Cutting
# concurrent workers slows the accumulation rate directly; resume-skip means only
# the still-missing runs (b200_highres128/training_9374, random_init) rerun.
export LP_DATASETS="Dataset001_CZII_10001_patches512"
export LP_NUM_WORKERS=8
bash /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/slurm_scripts/train_3dino_lp_h100_b200_comparison.sh
