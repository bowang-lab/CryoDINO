#!/bin/bash
#SBATCH -J subtomogram-generation
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

cd /cluster/home/$USER/scripts/cryoet/CryoET/preprocessing

# python subtomograms_generation.py \
#   -i ${CRYOET_ROOT}/datasets/10442_imagesTr \
#   -o ${CRYOET_ROOT}/datasets/10442_imagesTr_subtomograms \
#   -m ${CRYOET_ROOT}/datasets/otsu_masks/otsu_deconv

# python subtomograms_generation.py \
#   -i ${CRYOET_ROOT}/datasets/10443_imagesTr \
#   -o ${CRYOET_ROOT}/datasets/10443_imagesTr_subtomograms \
#   -m ${CRYOET_ROOT}/datasets/otsu_masks/otsu_deconv

NEW_BASE="${CRYOET_ROOT}/datasets/upload/nifti_unannotated_10476_10479_10493_06232026"
SUBTOMO_BASE="${CRYOET_ROOT}/datasets/nifti_unannotated_10476_10479_10493_06232026_subtomograms"

python subtomograms_generation.py \
  -i "${NEW_BASE}/10476/imagesTr" \
  -o "${SUBTOMO_BASE}/10476_subtomograms"

python subtomograms_generation.py \
  -i "${NEW_BASE}/10479/imagesTr" \
  -o "${SUBTOMO_BASE}/10479_subtomograms"

python subtomograms_generation.py \
  -i "${NEW_BASE}/10493/imagesTr" \
  -o "${SUBTOMO_BASE}/10493_subtomograms"