# Detection Pipeline

Run steps in order. `sbatch` submits the SLURM job; direct `python` forms are shown
for interactive runs.

| # | Step | Command |
|---|------|---------|
| 1 | Inspect all detection datasets → per-dataset stats JSON | `bash slurm_scripts/inspect_detection_datasets.sh` |
| 2 | Sanity-check raw train tomograms (440) — CSV blobs on slices | `sbatch slurm_scripts/visualize_detection_tomogram_overlay_czi.sh` |
| 2b | Same overlay check for val (445) | `sbatch slurm_scripts/visualize_detection_tomogram_overlay_czi_val.sh` |
| 3 | Patchify: raw NIfTI + CSV → 128³ `.pt` patches + `czi_100_datalist.json` | `sbatch slurm_scripts/patchify_detection_czi.sh` |
| 4 | Sanity-check generated patches (GT overlaid, 1 PNG/patch) | `sbatch slurm_scripts/visualize_detection_patches_czi.sh` |
| 5 | Fine-tune 3DINO (ViTAdapterUNETR); writes `results.json` + checkpoints | `sbatch slurm_scripts/train_3dino_ft_h100_detection_czi.sh` |
| 6 | Inference over full tomograms of 440/445/446 | `sbatch slurm_scripts/infer_3dino_ft_h100_detection_czi.sh` |

## Inference (step 6)

Runs `best_model.pth` over **full tomograms** of all three CZI datasets and writes to the
training `OUTPUT_DIR`:

| Dataset | Role | GT | Outputs |
|---------|------|----|---------|
| Dataset440_CZII_10440 | train | yes | `predictions_<name>.csv` + `results_inference_<name>.json` |
| Dataset445_CZII_10445 | val | yes | `predictions_<name>.csv` + `results_inference_<name>.json` |
| Dataset446_CZII_10446 | hidden test | no | `predictions_<name>.csv` only |

GT (+ per-run voxel size) is read from each dir's `point_annotations.csv` if present; 446 has none,
so it falls back to `--default-voxel-size` (10.0 Å) and produces predictions only.

CSV columns: `tomo_id, class_id, particle_type, x/y/z_vox, x/y/z_ang, score`.

### Direct invocation

Model build args MUST match the training run. `--checkpoint` defaults to `{output-dir}/best_model.pth`.

```bash
# Mode A (tomo-level, recommended): scan a raw Dataset*** dir's imagesTr/*.nii.gz.
PYTHONPATH=. python dinov2/eval/detection3d_inference.py \
    --config-file dinov2/configs/train/vit3d_highres.yaml \
    --output-dir /path/to/finetuning_detection/<run_dir> \
    --pretrained-weights /path/to/teacher_checkpoint.pth \
    --dataset-name czi --segmentation-head ViTAdapterUNETR --image-size 128 \
    --num-workers 10 --cache-dir /tmp/cryodino_infer_cache --resize-scale 1.0 \
    --raw-dataset-dir /path/to/czi_dataset/Dataset446_CZII_10446 \
    --run-name Dataset446_CZII_10446 --default-voxel-size 10.0 \
    --overlap 0.75 --min-score 0.05 --nms-iou-threshold 0.8

# Mode B: a split of the patch datalist. NOTE its `training` split is 128³ PATCHES,
# not full tomograms — use Mode A for tomo-level. Add --base-data-dir and --split {training,validation,test}.
```

> **Class-ordering caveat:** the model uses alphabetical class IDs, but `CZIDetectionMetrics` uses
> the official Kaggle order. The aggregate F-beta still matches training's reported value, but the
> metric's per-class radii/weights are permuted. Prediction CSVs use the correct alphabetical
> particle names. Fix `metrics.py` class order if you need a correct per-class breakdown.

## Other interactive scripts

```bash
# Raw-tomogram annotation overlay (step 2)
python visualization/detection_tomogram_overlay.py \
    --images-dir <Dataset440>/imagesTr \
    --csv        <Dataset440>/point_annotations.csv \
    --output-dir <vis_out> \
    --sigmas-ang '{"Beta-amylase":65,"Beta-galactosidase":90,"Thyroglobulin":130,"cytosolic ribosome":150,"ferritin complex":60,"virus-like capsid":135}' \
    --slice-tol 3 --n-tomos -1

# Generated-patch GT overlay (step 4)
python visualization/visualize_detection_patches.py \
    --datalist   <patches_dir>/czi_100_datalist.json \
    --output-dir <patches_dir>/Dataset440_CZII_10440_detection_patch_vis \
    --split training --n-patches -1 --slice-tol 5 --only-particles
```
