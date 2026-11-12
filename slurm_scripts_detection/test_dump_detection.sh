#!/bin/bash
#SBATCH -J det-testdump
#SBATCH -p gpu_pmccaiteam
#SBATCH -t 0-12:00:00
#SBATCH --account=pmccaiteam_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=36G
#SBATCH --output=/cluster/home/t129616uhn/projects/logs/%x_%j.log

# Re-run the test set of a finished detection run with its best_model.pth and save per-tomogram
# predictions (<run>/test_eval/test_predictions.pkl) for bootstrap CIs. Training outputs untouched.
#   RUN_NAME  run folder under finetuning_detection
#   HEAD      ViTAdapterUNETR | ResNetFPN
#   PERCENT   100 | 50 | 10   (CZII)
#   PRETRAINED=b200|none, DECODER_FUSION (default adapter), LORA_RANK (default 0)
set -u
: "${RUN_NAME:?}" "${HEAD:?}" "${PERCENT:?}"
PRETRAINED="${PRETRAINED:-none}"; DECODER_FUSION="${DECODER_FUSION:-adapter}"; LORA_RANK="${LORA_RANK:-0}"

date; hostname; nvidia-smi --query-gpu=name --format=csv,noheader
set +u; source ~/.bashrc; conda activate cryoet
# B200 (sm_100): torch 2.7.1 venv layered on cryoet, as in train_detection_hyp.sh
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q B200 && source /cluster/projects/pmccaiteam/sumin/envs/cryoet-b200/bin/activate
set -u
echo "Python env: $(python -c 'import sys, torch; print(sys.prefix, "torch", torch.__version__)')"
cd /cluster/home/t129616uhn/projects/CryoDINO/3DINO || exit 1

DETECTION_ROOT="/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets/downstream_detection"
EXPERIMENTS="/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments"
RUN_DIR="${EXPERIMENTS}/finetuning_detection/${RUN_NAME}"
PATCH_DIR="${DETECTION_ROOT}/Dataset440_CZII_10440_detection_patches_128"
case "$PERCENT" in 100) BASE_DATA_DIR="$PATCH_DIR" ;; 50|10) BASE_DATA_DIR="${PATCH_DIR}/czi_${PERCENT}percent" ;; *) exit 1 ;; esac
case "$PRETRAINED" in
  b200) WEIGHTS="${EXPERIMENTS}/ssl3d_run_b200_high_res_128/eval/training_6249/teacher_checkpoint.pth" ;;
  none) WEIGHTS="" ;;
esac
TEST_IMAGES="${TEST_IMAGES:-}"
OUT_DIR="${RUN_DIR}/${OUT_SUBDIR:-test_eval}"
CACHE_DIR="${EXPERIMENTS}/cache_dir_downstream_detection/testdump_${RUN_NAME}_${SLURM_JOB_ID}"
[ -f "$RUN_DIR/best_model.pth" ] || { echo "Missing $RUN_DIR/best_model.pth"; exit 1; }
[ -e "$OUT_DIR/test_predictions.pkl" ] && { echo "Exists: $OUT_DIR/test_predictions.pkl"; exit 1; }
rm -rf "$CACHE_DIR"; mkdir -p "$CACHE_DIR" "$OUT_DIR"

EXTRA=()
[ "$LORA_RANK" -gt 0 ] && EXTRA+=(--lora-rank "$LORA_RANK")
[ -n "$TEST_IMAGES" ] && EXTRA+=(--test-images "$TEST_IMAGES")
[ -n "${PATCH_SIZE_XYZ:-}" ] && EXTRA+=(--patch-size-xyz "$PATCH_SIZE_XYZ")
echo "Run: $RUN_DIR  head=$HEAD fusion=$DECODER_FUSION lora=$LORA_RANK weights=${WEIGHTS:-<none>}"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python dinov2/eval/detection3d_test_dump.py \
  --run-dir "$RUN_DIR" \
  --config-file dinov2/configs/train/vit3d_highres.yaml \
  --output-dir "$OUT_DIR" \
  --pretrained-weights "$WEIGHTS" \
  --dataset-name czi --dataset-percent 100 --base-data-dir "$BASE_DATA_DIR" \
  --segmentation-head "$HEAD" --decoder-fusion "$DECODER_FUSION" \
  --image-size 128 --batch-size 4 --num-workers 4 --cache-dir "$CACHE_DIR" --resize-scale 1.0 \
  ${EXTRA[@]+"${EXTRA[@]}"}
STATUS=$?
rm -rf "$CACHE_DIR"
date
exit $STATUS
