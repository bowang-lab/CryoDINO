#!/bin/bash
#SBATCH -J ssl-3dino-pretrain-high_res_h100
#SBATCH -p gpu_bwanggroup
#SBATCH -t 2-00:00:00
#SBATCH --account=bwanggroup_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=62
#SBATCH --mem=450G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

# for a100 GPUs:
# #SBATCH --cpus-per-task=120
# #SBATCH --mem=6000G
# nums of workers: 28


date
hostname
pwd
nvidia-smi
# =========================
# Environment
# =========================
source ~/.bashrc
conda activate cryoet

# =========================
# Paths
# =========================
cd /cluster/home/t139212uhn/scripts/cryoet/CryoET/3DINO || exit 1

CONFIG_FILE="dinov2/configs/ssl3d_default_config.yaml"
CONFIG_FILE_HIGH_RES="dinov2/configs/train/vit3d_highres.yaml"
OUTPUT_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_01_h100"
OUTPUT_DIR_HIGH_REZ="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100_high_res"
CACHE_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/cache_dir"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR_HIGH_REZ"
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

# PYTHONPATH=. \
# python -m torch.distributed.launch \
#   --nproc_per_node=${NUM_GPUS} \
#   --master_port=${MASTER_PORT} \
#   dinov2/train/train3d.py \
#   --config-file "${CONFIG_FILE}" \
#   --output-dir "${OUTPUT_DIR}" \
#   --cache-dir "${CACHE_DIR}"
echo "Pretraining job finished"

PYTHONPATH=. \
python -m torch.distributed.launch \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=${MASTER_PORT} \
  dinov2/train/train3d.py \
  --config-file "${CONFIG_FILE_HIGH_RES}" \
  --output-dir "${OUTPUT_DIR_HIGH_REZ}" \
  --cache-dir "${CACHE_DIR}"
echo "High-resolution pretraining job finished"

date

