#!/bin/bash
#SBATCH -J convert_b2nd
#SBATCH -p veryhimem
#SBATCH -t 1-00:00:00
#SBATCH --account=pmccaiteam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
##SBATCH --mail-user=<YOUR_EMAIL>
##SBATCH --mail-type=ALL
#SBATCH --output=slurm_logs/%x_%j.log

: "${CRYOET_ROOT:?set CRYOET_ROOT to the CryoDINO data folder (holding datasets/ and experiments/), e.g. export CRYOET_ROOT=/path/to/cryoet}"
date
hostname
pwd

source ~/.bashrc
conda activate cryoet

cd /cluster/home/$USER/scripts/cryoet/CryoET/dataset/downstream

python convert_to_b2nd.py \
  "${CRYOET_ROOT}/experiments/Dataset001_CZII_10001_100_datalist.json"

python convert_to_b2nd.py \
  "${CRYOET_ROOT}/experiments/Dataset010_CZII_10010_100_datalist.json"

python convert_to_b2nd.py \
  "${CRYOET_ROOT}/experiments/Dataset989_EMPIAR_10989_transposed_100_datalist.json"