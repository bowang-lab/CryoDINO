#!/bin/bash
#SBATCH -J subtomogram-generation
#SBATCH -p veryhimem
#SBATCH -t 1-00:00:00
#SBATCH --account=bwanggroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

date
hostname
pwd

source ~/.bashrc
conda activate cryoet

cd /cluster/home/t139212uhn/scripts/cryoet/CryoET/preprocessing

python subtomograms_generation.py \
  -i /cluster/projects/bwanggroup/reza/projects/cryoet/datasets/10442_imagesTr \
  -o /cluster/projects/bwanggroup/reza/projects/cryoet/datasets/10442_imagesTr_subtomograms \
  -m /cluster/projects/bwanggroup/reza/projects/cryoet/datasets/otsu_masks/otsu_deconv

python subtomograms_generation.py \
  -i /cluster/projects/bwanggroup/reza/projects/cryoet/datasets/10443_imagesTr \
  -o /cluster/projects/bwanggroup/reza/projects/cryoet/datasets/10443_imagesTr_subtomograms \
  -m /cluster/projects/bwanggroup/reza/projects/cryoet/datasets/otsu_masks/otsu_deconv