#!/bin/bash
#SBATCH -J 3dino-infer-detection-czi
#SBATCH -p gpu_bwanggroup
#SBATCH -t 0-08:00:00
#SBATCH --account=bwanggroup_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --mail-user=sum.kim@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t129616uhn/projects/logs/%x_%j.log

# =========================
# Inference: 3DINO detection (CZI) — standalone, no training.
# Runs sliding-window detection over FULL tomograms of all three CZI datasets:
#   Dataset440_CZII_10440  → train  tomograms (GT present  → predictions + F-beta)
#   Dataset445_CZII_10445  → val    tomograms (GT present  → predictions + F-beta)
#   Dataset446_CZII_10446  → hidden test       (no GT      → predictions only)
#
# Each dataset is scanned at imagesTr/*.nii.gz; GT (+ per-run voxel_size) is read
# from <dir>/point_annotations.csv when present. Outputs land in OUTPUT_DIR:
#   predictions_<DatasetName>.csv          (always)
#   results_inference_<DatasetName>.json   (only when GT present)
#
# Model build args MUST match train_3dino_ft_h100_detection_czi.sh so the head is
# rebuilt identically before loading the fine-tuned weights (best_model.pth).
# =========================

date
hostname
pwd
nvidia-smi

source ~/.bashrc
conda activate cryoet

cd /cluster/home/t129616uhn/projects/CryoDINO/3DINO || exit 1

# =========================
# Paths
# =========================
DETECTION_BASE="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/downstream_detection"
CZI_DATASET_DIR="${DETECTION_BASE}/czi_dataset"
BASE_DATA_DIR="${DETECTION_BASE}/Dataset440_CZII_10440_detection_patches_128"   # only for model-arg parity
BASE_OUTPUT_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/datasets/finetuning_detection"

# =========================
# Fixed Parameters (must match training)
# =========================
CONFIG_FILE="dinov2/configs/train/vit3d_highres.yaml"
PRETRAINED_WEIGHTS="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100_high_res/eval/training_9374/teacher_checkpoint.pth"
DATASET_NAME="czi"            # czi → 6 classes
DATASET_PERCENT=100
SEGMENTATION_HEAD="ViTAdapterUNETR"
IMAGE_SIZE=128                # must match training image-size / patch size
BATCH_SIZE=4                  # unused in raw mode; kept for arg parity
NUM_WORKERS=10
RESIZE_SCALE=1.0

# CZI 6-class sigma map (radius Å); only used for GT sigma bookkeeping
CZI_SIGMAS_ANG='{"Beta-amylase":65,"Beta-galactosidase":90,"Thyroglobulin":130,"cytosolic ribosome":150,"ferritin complex":60,"virus-like capsid":135}'

# =========================
# Inference-specific
# =========================
OVERLAP=0.75
MIN_SCORE=0.05
NMS_IOU=0.8
DEFAULT_VOXEL_SIZE=10.0       # used for tomograms with no CSV row (hidden test 10446)

OUTPUT_DIR="${BASE_OUTPUT_DIR}/ssl3d_run_h100_high_res_training_9374_${DATASET_NAME}_detection_vit_adapter"
CHECKPOINT="${OUTPUT_DIR}/best_model.pth"

mkdir -p "$OUTPUT_DIR"

# Dataset dirs to run over (train, val, hidden test)
DATASETS=(
  "Dataset440_CZII_10440"
  "Dataset445_CZII_10445"
  "Dataset446_CZII_10446"
)

echo "=============================================="
echo "3D detection inference over all CZI datasets"
echo "Checkpoint     : $CHECKPOINT"
echo "Output dir     : $OUTPUT_DIR"
echo "Datasets       : ${DATASETS[*]}"
echo "=============================================="

for DS in "${DATASETS[@]}"; do
  RAW_DIR="${CZI_DATASET_DIR}/${DS}"
  echo ""
  echo "----- Inference on ${DS} (${RAW_DIR}) -----"

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python dinov2/eval/detection3d_inference.py \
    --config-file "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --pretrained-weights "$PRETRAINED_WEIGHTS" \
    --checkpoint "$CHECKPOINT" \
    --dataset-name "$DATASET_NAME" \
    --dataset-percent "$DATASET_PERCENT" \
    --base-data-dir "$BASE_DATA_DIR" \
    --segmentation-head "$SEGMENTATION_HEAD" \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --cache-dir "/tmp/cryodino_infer_cache" \
    --resize-scale "$RESIZE_SCALE" \
    --raw-dataset-dir "$RAW_DIR" \
    --run-name "$DS" \
    --sigmas-ang "$CZI_SIGMAS_ANG" \
    --default-voxel-size "$DEFAULT_VOXEL_SIZE" \
    --overlap "$OVERLAP" \
    --min-score "$MIN_SCORE" \
    --nms-iou-threshold "$NMS_IOU"

  echo "----- Done ${DS}: ${OUTPUT_DIR}/predictions_${DS}.csv -----"
done

echo ""
echo "Finished inference on all datasets."
echo "  predictions_Dataset44{0,5,6}*.csv  (+ results_inference_*.json for 440/445)"
date
