#!/bin/bash
#SBATCH -J cryodino_lp-ds989
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

# One job per dataset: ALL backbones/checkpoints for EMPIAR-10989 only.
# This job exclusively owns the 10989 cache dir -> no cross-job cache race.
export LP_DATASETS="Dataset989_EMPIAR_10989_transposed_patches512"
bash /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/slurm_scripts/train_3dino_lp_h100_b200_comparison.sh
