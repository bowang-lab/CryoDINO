# Particle Detection (CZI) — full-tomogram val/test

Fine-tunes a CryoDINO backbone with a detection head (`3DINO/dinov2/eval/detection3d.py`). The head is anchor-free, with 6 classes and stride 2.

- **Training** uses pre-extracted 128³ `.pt` patches.
- **Validation and test** run a sliding window over each **whole tomogram** and score it against that tomogram's global GT. Each particle is counted exactly once, and the score is the Kaggle CZII F4.

Two steps: **build a datalist json**, then **submit the SLURM job** with that json.

---

## 1. Datalist json

```json
{
  "training":   [{"image": ".../images/TS_5_4_0_64_56.pt",
                  "points": [[x, y, z, class_id, sigma_vox], ...],   // patch-local voxels
                  "voxel_size": 10.012}, ...],
  "validation": [{"image": ".../imagesTr/TS_69_2_0000.nii.gz",
                  "points": [[x, y, z, class_id, sigma_vox], ...],   // global voxels, whole tomogram
                  "voxel_size": 10.012}],
  "test":       [{"image": ".../imagesTr/TS_73_6_0000.nii.gz", "points": [...], "voxel_size": 10.012}]
}
```

- Coordinates are **XYZ in voxels**, in nibabel axis order. The CSV stores `z, y, x`, and the scripts reorder it.
- `class_id` is 0-based, assigned alphabetically by particle name: 0 Beta-amylase, 1 Beta-galactosidase, 2 Thyroglobulin, 3 cytosolic ribosome, 4 ferritin complex, 5 virus-like capsid.
- A row with `class_id = -100` is padding (a patch with no particles).
- Val/test may hold several tomograms each. The F4 threshold sweep runs over all of them together.

### Build it

The `.pt` patches and their patch-level datalist come from `preprocessing/downstream_patch_generation.py --detection` (see `slurm_scripts/patchify_detection_czi.sh`). Then split by tomogram, with val/test as full tomograms:

```bash
python preprocessing/make_detection_split.py \
    --datalist-json <patches_dir>/czi_100_datalist.json \
    --images-dir    <patches_dir>/images \
    --tomo-dir      <Dataset440_CZII_10440>/imagesTr \
    --csv           <Dataset440_CZII_10440>/point_annotations.csv \
    --val-tomo TS_69_2 --test-tomo TS_73_6 \
    --output-json   <out>/czi_fulltomo_datalist.json
```

- Only the source json's `training` split is read, because it is the only split with GT.
- `--images-dir` rewrites patch paths for the machine you are on.
- The script asserts three things:
  - no tomogram appears in more than one split
  - val/test tomograms exist
  - every training-patch point plus its patch offset matches a CSV point. This catches axis-order or offset mismatches that would otherwise silently wreck the F4.
- Without `--tomo-dir/--csv`, val/test stay as patches. That is the old behaviour and it over-counts particles in the overlap zones.

---

## 2. Train + evaluate (SLURM)

```bash
# check the json first (no GPU, a few seconds)
PREFLIGHT_ONLY=1 bash slurm_scripts/train_3dino_ft_detection_czi_fulltomo.sh <out>/czi_fulltomo_datalist.json

# submit
sbatch slurm_scripts/train_3dino_ft_detection_czi_fulltomo.sh <out>/czi_fulltomo_datalist.json [OUTPUT_DIR]
```

Before training, the job checks the datalist: all splits are non-empty, patch and tomogram files exist, val/test are `.nii.gz`, and each has GT. If any check fails it exits immediately.

To override hyperparameters, set environment variables (defaults in brackets):

| Variable | Default | |
|---|---|---|
| `EPOCHS`, `EPOCH_LENGTH` | 100, 300 | total iters = product |
| `EVAL_ITERS` | 1500 | a full-tomogram val pass every N iters |
| `WARMUP_ITERS` | 3000 | |
| `BATCH_SIZE`, `LEARNING_RATE` | 4, 1e-4 | |
| `SEGMENTATION_HEAD` | ViTAdapterUNETR | also `UNETR`, `Linear` |
| `PRETRAINED_WEIGHTS` | b200 highres128 / training_6249 teacher | |
| `NUM_WORKERS`, `DATASET_PERCENT` | 10, 100 | |

Example of a short run: `EPOCHS=10 EPOCH_LENGTH=202 EVAL_ITERS=505 WARMUP_ITERS=200 sbatch ... <json>`.

Paths such as the repo location, config, output and cache base, and conda env are set at the top of the script for the `bwanggroup` cluster layout. Edit them if yours differ. The backbone is frozen: the script does not pass `--train-feature-model`.

To run without SLURM, call `detection3d.py` with the same flags as the script, plus `--datalist-json <json>`.

### Outputs (in `OUTPUT_DIR`)

- `results.json` contains:
  - `iters_list`, `train_loss_list`
  - `val_f4_list`, `val_per_cls_f4_list`
  - `test_f4`, `test_per_cls_f4`
- `best_model.pth` is the checkpoint with the best val F4, and it is used for the test pass.
- `model_iter*_f4*.pth` are the top-5 checkpoints by val F4.
- The log prints per-class F4 and the per-class score thresholds chosen by the sweep.

Test uses sliding-window overlap 0.75 and val uses 0.5. One 630×630×184 tomogram is 867 tiles at test and 162 at val.

### Reference numbers

Dataset440, 5 train tomograms, val TS_69_2 (143 particles), test TS_73_6 (216 particles). This was a short run: 2020 iters, frozen backbone, ViTAdapterUNETR.

| | F4 |
|---|---|
| Val (best, iter 2020) | 0.655 |
| Test | 0.604 |

These numbers are not directly comparable to patch-based val/test scores, which count overlapping particles several times. Each test tomogram has only 12–95 particles per class, so single-run numbers are noisy. Beta-galactosidase is the weakest class.

---

## Troubleshooting

- **`Exception: Could not deserialize ATN with version 3 (expected 4)`** at import: the env has the wrong antlr4 runtime for `omegaconf 2.3.0`. Fix with `pip install antlr4-python3-runtime==4.9.3`, which is the version pinned in `requirements*.txt` and `environment_cryodino.yml`.
- **"A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"** traceback at startup: this is only a warning from an optional pandas/bottleneck import, and training continues.
- **Val/test F4 is 0.0 for every class**: the val/test entries have no GT. The shipped `czi_100_datalist.json` is like this. Rebuild the json with `make_detection_split.py --tomo-dir --csv`.
