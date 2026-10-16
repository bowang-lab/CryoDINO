#!/bin/bash
# Submit the full DeepETPicker run (prepare -> preprocess -> train -> infer -> score) to SLURM.
# Edit the paths below (or export them), then:  sbatch submit.sbatch
#
# Sizing note: the dataset is small -- Dataset440 is ~900 particle-centred training samples per
# epoch at batch_size 8 (~113 steps), and the whole preprocessing step is a few minutes of CPU.
# A 3g.40gb MIG slice is plenty; 4 h covers 60 epochs with margin.
#SBATCH --job-name=deepetpicker
#SBATCH --account=rrg-wanglab-ab
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1   # or --gres=gpu:1 on non-MIG clusters
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1:30:00   # measured: 31 s/epoch -> 60 epochs ~= 31 min, +1 min preprocess
                         # +2.5 min inference ~= 35 min.  Keep the request tight: a 4 h ask
                         # is not backfill-eligible and sat in the queue for over an hour.
#SBATCH --output=/scratch/cyyu/cryodino/logs/%x-%j.out   # not the kit dir -- keep it clean
set -euo pipefail

# NOT $0/BASH_SOURCE: sbatch copies this script to a per-job spool dir on the compute node
# (/localscratch/spool/slurmd/job<ID>/), so deriving the kit path from it finds nothing.
KIT="${KIT:-/home/cyyu/projects/def-wanglab-ab/cyyu/deepetpicker_kit}"
[ -f "$KIT/run_datalist.sh" ] || { echo "ERROR: no run_datalist.sh under KIT=$KIT -- set KIT=<kit dir>"; exit 1; }
export VENV="${VENV:-/scratch/cyyu/venvs/deepetpicker}"
export DEP="${DEP:-/scratch/cyyu/cryodino/DeepETPicker}"
export EPOCHS="${EPOCHS:-60}"

DATALIST="${DATALIST:-/home/cyyu/projects/def-wanglab-ab/cryodino/Dataset440_CZII_10440_detection_patches_128/czi_100_datalist.json}"
OUT="${OUT:-/scratch/cyyu/cryodino/deepetpicker_ds440}"

module load python/3.11 2>/dev/null || true
nvidia-smi -L || true

bash "$KIT/run_datalist.sh" "$DATALIST" "$OUT"

# $OUT lives on scratch, which is purged. Copy the small artefacts somewhere durable:
#   PRED=$(dirname $(dirname $(ls -t $OUT/runs/*/*/version_*/checkpoints/*.ckpt | head -1)))
#   cp -r $PRED/PredictedLabels /home/cyyu/projects/def-wanglab-ab/cryodino/baselines_out/deepetpicker_ds440/
