#!/bin/bash
#SBATCH -J cryodino_3dino-lp-cmp-job4-hr128rand
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

# Job 4: B200 high-res-128 (all eval iterations) + random-init baseline x 3 datasets.
bash /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/slurm_scripts/train_3dino_lp_h100_b200_comparison.sh \
    b200_highres128 random_init
