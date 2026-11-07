#!/bin/bash
#SBATCH -J score-heads
#SBATCH -p himem
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --output=slurm-%x-%j.out
set -eo pipefail
source /cluster/home/t149164uhn/miniconda3/etc/profile.d/conda.sh
conda activate cryodino

MODEL="${MODEL:?Submit with MODEL=highres128 or MODEL=highres112}"
D="/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets"
cd /cluster/home/t149164uhn/projects/CryoDINO
python visualization/score_heads.py \
  --vis-base "$HOME/projects/cryoet_pca/vis_output_336" --model "$MODEL" \
  --item "TS_0001=$D/downstream/Dataset001_CZII_10001/labelsTr/TS_0001.nii.gz" \
  --item "TS_0009=$D/downstream/Dataset001_CZII_10001/labelsTr/TS_0009.nii.gz" \
  --item "TS_027=$D/downstream/Dataset000_CZII_10000/labelsTr/TS_027.nii.gz" \
  --item "TS_100_3=$D/downstream_detection/czi_dataset/Dataset445_CZII_10445/labelsTr/TS_100_3.nii.gz" \
  --item "TS_100_4=$D/downstream_detection/czi_dataset/Dataset446_CZII_10446/labelsTr/TS_100_4.nii.gz" \
  --item "tomo_01a877=$D/downstream_detection/byu_dataset/labelsTr/tomo_01a877.nii.gz"
