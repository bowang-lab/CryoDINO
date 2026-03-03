#!/bin/bash
#SBATCH -J 3dino-vis-attn-maps
#SBATCH -p gpu_bwanggroup
#SBATCH -t 1-00:00:00
#SBATCH --account=bwanggroup_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --mail-user=attarpour1993@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/cluster/home/t139212uhn/scripts/cryoet/slurm_logs/%x_%j.log

# =========================
# Visualisation: PCA + MHSA for 3 representative tomograms
#   TS_0001  → train sample       (Dataset001, CZII 10001)
#   TS_0009  → internal val test  (Dataset001, CZII 10001)
#   TS_027   → external val test  (Dataset001, CZII 10001)
# Preprocessing: z-score whole tomo → CropForeground → patchwise
#   ScaleIntensityRangePercentiles(0.5-99.5) inside sliding window
# =========================

date
hostname
nvidia-smi

source ~/.bashrc
conda activate cryoet

cd /cluster/home/t139212uhn/scripts/cryoet/CryoET/3DINO || exit 1

# =========================
# Fixed parameters
# =========================
CONFIG="dinov2/configs/train/vit3d_highres.yaml"
WEIGHTS="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/ssl3d_run_h100_high_res/eval/training_9374/teacher_checkpoint.pth"
IMG_DIR="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/attention_maps/imgs"
OUT_BASE="/cluster/projects/bwanggroup/reza/projects/cryoet/experiments/attention_maps/vis_output"
IMAGE_SIZE=112

SAMPLES=("TS_0001_0000" "TS_0009_0000" "TS_027_0000")
NAMES=("TS_0001"        "TS_0009"       "TS_027")

for i in "${!SAMPLES[@]}"; do
    IMG="${IMG_DIR}/${SAMPLES[$i]}.nii.gz"
    NAME="${NAMES[$i]}"

    for VIS_TYPE in pca mhsa; do
        OUT_DIR="${OUT_BASE}/${NAME}/${VIS_TYPE}"
        mkdir -p "$OUT_DIR"

        echo "=============================================="
        echo "  ${NAME}  |  vis-type: ${VIS_TYPE}"
        echo "  Output → ${OUT_DIR}"
        echo "=============================================="

        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        PYTHONPATH=. python dinov2/eval/vis_pca_cryoet.py \
            --config-file        "$CONFIG"      \
            --pretrained-weights "$WEIGHTS"     \
            --image-path         "$IMG"         \
            --output-dir         "$OUT_DIR"     \
            --vis-type           "$VIS_TYPE"    \
            --input-type         sliding_window \
            --image-size         "$IMAGE_SIZE"

        echo "Done: ${OUT_DIR}"
        echo ""
    done
done

echo "All visualisations complete!"
date
