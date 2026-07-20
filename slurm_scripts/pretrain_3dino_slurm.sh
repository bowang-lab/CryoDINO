#!/bin/bash
#SBATCH -J ssl-3dino-pretrain-high_res_b200
#SBATCH -p gpu_pmcc_ai_team
#SBATCH -t 7-00:00:00
#SBATCH --account=pmcc_ai_team_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=800G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

# for a100 GPUs: --cpus-per-task=120, --mem=6000G, num_workers=28
# for b200 GPUs: --cpus-per-task=120, --mem=450G, num_workers=28


date
hostname
pwd
nvidia-smi
# =========================
# Environment
# =========================
source ~/.bashrc
conda activate cryodino

# =========================
# Paths
# =========================
cd /cluster/home/t139212uhn/scripts/cryoet/CryoDINO/3DINO || exit 1

CONFIG_FILE="dinov2/configs/ssl3d_default_config.yaml"
CONFIG_FILE_HIGH_RES="dinov2/configs/train/vit3d_highres.yaml"
OUTPUT_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_b200"
OUTPUT_DIR_HIGH_REZ="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_b200_high_res"
OUTPUT_DIR_HIGH_REZ_128="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_b200_high_res_128"
CACHE_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/cache_dir"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR_HIGH_REZ"
mkdir -p "$OUTPUT_DIR_HIGH_REZ_128"
mkdir -p "$CACHE_DIR"

# =========================
# Distributed Training Vars
# =========================
export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MASTER_PORT=29501
NUM_GPUS=4

# =========================
# Run 3DINO SSL Pretraining
# =========================
echo "Starting 3DINO SSL pretraining"
echo "Config: $CONFIG_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Cache dir: $CACHE_DIR"

# --- Stage 1: SSL pretraining (96^3) — DONE, commented out ---
# PYTHONPATH=. python -m torch.distributed.launch \
#   --nproc_per_node=${NUM_GPUS} \
#   --master_port=${MASTER_PORT} \
#   dinov2/train/train3d.py \
#   --config-file "${CONFIG_FILE}" \
#   --output-dir "${OUTPUT_DIR}" \
#   --cache-dir "${CACHE_DIR}" || exit 1
# echo "Pretraining job finished"

# --- Stage 2: high-res adaptation (112^3) — DONE, commented out ---
# PYTHONPATH=. python -m torch.distributed.launch \
#   --nproc_per_node=${NUM_GPUS} \
#   --master_port=${MASTER_PORT} \
#   dinov2/train/train3d.py \
#   --config-file "${CONFIG_FILE_HIGH_RES}" \
#   --output-dir "${OUTPUT_DIR_HIGH_REZ}" \
#   --cache-dir "${CACHE_DIR}"
# echo "High-resolution pretraining job finished"

# --- Stage 2b: high-res adaptation (128^3, batch 130) ---
PYTHONPATH=. python -m torch.distributed.launch \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=${MASTER_PORT} \
  dinov2/train/train3d.py \
  --config-file "${CONFIG_FILE_HIGH_RES}" \
  --output-dir "${OUTPUT_DIR_HIGH_REZ_128}" \
  --cache-dir "${CACHE_DIR}"
echo "High-resolution (128) adaptation job finished"

date

