#!/bin/bash
#SBATCH -J convert_b2nd
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

cd /cluster/home/t139212uhn/scripts/cryoet/CryoET/dataset/downstream

python convert_to_b2nd.py \
  /cluster/projects/bwanggroup/reza/projects/cryoet/experiments/Dataset001_CZII_10001_100_datalist.json

python convert_to_b2nd.py \
  /cluster/projects/bwanggroup/reza/projects/cryoet/experiments/Dataset010_CZII_10010_100_datalist.json

python convert_to_b2nd.py \
  /cluster/projects/bwanggroup/reza/projects/cryoet/experiments/Dataset989_EMPIAR_10989_transposed_100_datalist.json