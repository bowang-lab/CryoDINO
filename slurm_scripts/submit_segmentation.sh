#!/bin/bash
# Submit the segmentation runs on H4H: one job per (preset or model, dataset, percent).
#
# Usage (from the CryoDINO checkout):
#   bash slurm_scripts/submit_segmentation.sh <what> ["<datasets>"] ["<percents>"] [--submit]
#     <what>      a train_segmentation.sh PRESET (h100_9374 | b200_112 | b200_128 | b200_best |
#                 b200_best_full | random_init), or
#                 baselines   UNet + UNETR (train_baselines.sh), or
#                 lp_compare  linear-probing sweep over pretraining checkpoints, one job per dataset
#                             (train_3dino_lp_h100_b200_comparison.sh; percents are ignored)
#     <datasets>  default "001 010 989 049"
#     <percents>  default "100"
#   Without --submit the sbatch commands are only printed.
#
# Examples:
#   bash slurm_scripts/submit_segmentation.sh b200_best "001 010 989 049" "50 10" --submit
#   bash slurm_scripts/submit_segmentation.sh baselines "" "100 50 10"
#
# Memory/time per run are the values the reported runs needed (Dataset001 is the largest).
# Extra environment for the jobs (e.g. DEEP_SUPERVISION=1, NUM_WORKERS=8) can be passed through
# EXTRA_EXPORT="VAR=value,VAR2=value".

set -euo pipefail
WHAT="${1:?usage: submit_segmentation.sh <preset|baselines|lp_compare> [\"<datasets>\"] [\"<percents>\"] [--submit]}"
DATASETS="${2:-}"; DATASETS="${DATASETS:-001 010 989 049}"
PERCENTS="${3:-}"; PERCENTS="${PERCENTS:-100}"
SUBMIT="${4:-}"
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
EXTRA="${EXTRA_EXPORT:+,$EXTRA_EXPORT}"

submit() {
  if [ "$SUBMIT" = "--submit" ]; then "$@"; else echo "$*"; fi
}

for DATASET in $DATASETS; do
  case "$WHAT" in
    lp_compare)
      case "$DATASET" in
        001) DS="Dataset001_CZII_10001_patches512";              MEM=700G; LPW=",LP_NUM_WORKERS=8" ;;  # OOMs with 16 workers
        010) DS="Dataset010_CZII_10010_patches512";              MEM=220G; LPW="" ;;
        989) DS="Dataset989_EMPIAR_10989_transposed_patches512"; MEM=220G; LPW="" ;;
        049) DS="Dataset049_EMPIAR_12049_transposed_patches512"; MEM=220G; LPW="" ;;
        *) echo "Unknown DATASET: $DATASET"; exit 1 ;;
      esac
      # one job per dataset, so each job owns its dataset's cache directory
      submit sbatch -J "cryodino_lp-ds${DATASET}" -t 7-00:00:00 --cpus-per-task=32 --mem="$MEM" \
        --export="ALL,LP_DATASETS=${DS}${LPW}${EXTRA}" "${SCRIPT_DIR}/train_3dino_lp_h100_b200_comparison.sh"
      continue ;;
  esac
  for PERCENT in $PERCENTS; do
    case "$WHAT" in
      baselines)
        for MODEL in unet unetr; do
          submit sbatch -J "seg-${MODEL}-${DATASET}-${PERCENT}" -t 2-00:00:00 --cpus-per-task=24 --mem=220G \
            --export="ALL,MODEL=${MODEL},DATASET=${DATASET},PERCENT=${PERCENT}${EXTRA}" "${SCRIPT_DIR}/train_baselines.sh"
        done ;;
      h100_9374)
        submit sbatch -J "seg-${WHAT}-${DATASET}-${PERCENT}" -t 2-00:00:00 --cpus-per-task=24 --mem=220G \
          --export="ALL,PRESET=${WHAT},DATASET=${DATASET},PERCENT=${PERCENT}${EXTRA}" "${SCRIPT_DIR}/train_segmentation.sh" ;;
      b200_112|b200_128|b200_best)
        MEM=400G; [ "$DATASET" = 001 ] && MEM=700G
        submit sbatch -J "seg-${WHAT}-${DATASET}-${PERCENT}" -t 2-00:00:00 --cpus-per-task=32 --mem="$MEM" \
          --export="ALL,PRESET=${WHAT},DATASET=${DATASET},PERCENT=${PERCENT}${EXTRA}" "${SCRIPT_DIR}/train_segmentation.sh" ;;
      b200_best_full|random_init)   # encoder is trained too: more memory and time
        submit sbatch -J "seg-${WHAT}-${DATASET}-${PERCENT}" -t 3-00:00:00 --cpus-per-task=32 --mem=800G \
          --export="ALL,PRESET=${WHAT},DATASET=${DATASET},PERCENT=${PERCENT}${EXTRA}" "${SCRIPT_DIR}/train_segmentation.sh" ;;
      *) echo "Unknown <what>: $WHAT"; exit 1 ;;
    esac
  done
done
