#!/bin/bash
# Submit the detection benchmark (CryoDINO / random-init / ResNetFPN) on H4H.
#
# Usage (from /cluster/home/t129616uhn/projects/CryoDINO):
#   bash slurm_scripts_detection/submit_detection.sh 100          # print the sbatch commands only
#   bash slurm_scripts_detection/submit_detection.sh 100 --submit # actually submit
#   bash slurm_scripts_detection/submit_detection.sh "50 10" --submit
#
# One job per (method, dataset, percent). BYU gets a longer wall time: its val/test are
# 97/98 whole tomograms run through sliding-window inference at every eval.

set -euo pipefail

PERCENTS="${1:?usage: submit_detection.sh \"<percents>\" [--submit]}"
SUBMIT="${2:-}"

SCRIPT_DIR="${SCRIPT_DIR:-/cluster/home/t129616uhn/projects/CryoDINO/slurm_scripts_detection}"
METHODS=(cryodino random_init resnetfpn)
DATASETS=(czi byu)

for PERCENT in $PERCENTS; do
  for DATASET in "${DATASETS[@]}"; do
    case "$DATASET" in
      czi) TIME="3-00:00:00" ;;
      byu) TIME="7-00:00:00" ;;
    esac
    for METHOD in "${METHODS[@]}"; do
      CMD=(sbatch
           -J "det-${METHOD}-${DATASET}-${PERCENT}"
           -t "$TIME"
           --export=ALL,METHOD=${METHOD},DATASET=${DATASET},PERCENT=${PERCENT}
           "${SCRIPT_DIR}/train_detection.sh")
      if [ "$SUBMIT" = "--submit" ]; then
        "${CMD[@]}"
      else
        echo "${CMD[*]}"
      fi
    done
  done
done
