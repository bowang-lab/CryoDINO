# DeePiCt 3D-CNN training pipeline (reusable for cryo-ET datasets)

Train a [DeePiCt](https://github.com/ZauggGroup/DeePiCt) 3D U-Net for voxel segmentation /
particle localization on any cryo-ET dataset, on an HPC GPU node. Wraps DeePiCt's own code with
a **fixed training schedule of `epochs` × `iterations_per_epoch`** (default **100 × 300**) so the
training budget is consistent regardless of dataset size.

> **Status:** validated end-to-end on the CZII Dataset010 patches (binary `particle`) on H100
> MIG nodes, no NaNs.
> - 2-tomogram smoke: val Dice 0.45 → 0.79; the `100×300` schedule mechanism verified (startup
>   assert passes).
> - **Full 68-tomogram run** (SLURM, `--mem=192G`): 11,822 boxes, `10×300` in **22 min**, peak
>   RAM **94 GB**, val Dice 0.45 → 0.63.
>
> The 100-epoch production run and the prediction/postprocessing stage still remain to run.

## What's here
| File | Role |
|---|---|
| `setup.sh` | Clone DeePiCt + apply modern-stack compatibility patches (torch/numpy/pandas). Idempotent. |
| `pt_to_mrc.py` | **Example** converter: CZII `.pt` cubes → `.mrc` + `metadata.csv`. Dataset-specific. |
| `prepare_data.sh` | **Example** data prep: convert → spectrum-match → `metadata.csv`. Dataset-specific. |
| `config.template.yaml` | Copy → edit → your run config. |
| `run_pipeline.sh` | **Generic**: partition training tomograms → train (100×300). Works for any dataset. |
| `deepict_train.py` | **Generic** trainer; enforces the 100×300 schedule via a replacement sampler. |

## One-time setup
Run everything **from inside this folder** — the scripts default to `./DeePiCt` and `./venv`, so
transferring just this `pipeline/` folder is enough (no paths point outside it). Override with
`DEEPICT=` / `VENV=` / `DATA=` if you want them elsewhere.
```bash
cd pipeline
bash setup.sh                       # clones DeePiCt -> ./DeePiCt and applies the patches
# Build the venv at ./venv (Digital Research Alliance / generic pip both work):
module load python/3.11             # or: module load python
python -m venv ./venv && source ./venv/bin/activate
python -m pip install torch torchvision monai mrcfile h5py tensorboardX pandas \
                      scikit-image scipy pyyaml tqdm matplotlib imageio nibabel
# On a Digital Research Alliance cluster you can add --no-index to pull prebuilt wheels faster.
```
The patches `setup.sh` applies (needed because DeePiCt targets Python 3.7):
`np.int→int` (numpy≥1.24), `ReduceLROnPlateau(verbose=)` removed (torch≥2.0),
`DataFrame.append→pd.concat` (pandas≥2.0).

## Run (on a GPU node with enough RAM — see Resources)
From inside `pipeline/`, with `./DeePiCt` and `./venv` in place (defaults):
```bash
bash prepare_data.sh                   # -> ./data/metadata.csv  (dataset-specific; edit for your data)
cp config.template.yaml myrun.yaml     # edit paths, training_list, semantic_classes
bash run_pipeline.sh myrun.yaml        # partition training tomograms -> train
```
(If your DeePiCt repo / venv / data live elsewhere, prefix with `DEEPICT=… VENV=… DATA=…`.)
Outputs: `output_dir/<model>.pth`, `_best.pth`, `_last.pth`; TensorBoard logs in
`output_dir/logging/`. Copy the model off `/scratch` (it gets purged).

### Cluster run (SLURM / sbatch)
The full `100×300` run is long and memory-hungry — submit it to a big-RAM GPU node instead of
running in a shell. Edit the placeholders in `submit.sbatch` (account, GPU, paths, config), then:
```bash
sbatch submit.sbatch            # runs prepare (optional) + run_pipeline.sh on the node
squeue --me                     # watch it; logs land in the --output path
```
`submit.sbatch` requests `--mem=192G` and one MIG GPU by default (see Resources for why).

## Preparing a NEW dataset (the only dataset-specific work)
Produce, for each tomogram you want to train on:
1. a raw tomogram `.mrc`, and
2. one binary `.mrc` mask **per semantic class** (voxel = 1 inside the target),

then a `metadata.csv` with columns:
```
tomo_name , tomo , <region_col> , <class>_mask
```
- `tomo` → raw `.mrc` path; `<class>_mask` → mask for each name in `semantic_classes`
  (e.g. class `particle` needs a `particle_mask` column).
- `<region_col>` (e.g. `lamella_file`) → an optional region mask, only used in postprocessing.

Point `dataset_table`, `tomos_sets.training_list`, and `semantic_classes` in the config at these.
`pt_to_mrc.py` / `prepare_data.sh` are the worked example for `.pt` cubes; swap them for your own
converter if your raw data is already `.mrc` or comes as coordinate lists (see DeePiCt's
`motl2sph_mask.py` to turn particle coordinates into spherical masks).

## The 100 × 300 schedule
Stock DeePiCt runs one pass over all partition boxes per epoch, so epoch length scales with the
data. `deepict_train.py` re-wraps the training `DataLoader` with a replacement `RandomSampler`
(`num_samples = batch_size × iterations_per_epoch`), giving **exactly `iterations_per_epoch`
iterations every epoch** for `epochs` epochs. Set both in the config:
`training.iterations_per_epoch: 300` and `training.unet_hyperparameters.epochs: 100`.
It asserts the loader length equals `iterations_per_epoch` at startup.

## One-shot from a datalist JSON (train → infer on test → save per sample)
Given a MONAI-style datalist `{"training":[…],"validation":[…],"test":[…]}` of
`{"image","label"}` entries (paths used **verbatim**; `.pt` cubes or `.nii.gz`/`.mrc` auto-detected,
labels binarised `>0`):
```bash
EPOCHS=100 ITERS=300 DEEPICT=/path/DeePiCt VENV=/path/venv bash run_datalist.sh datalist.json run_out
```
Runs `prepare_from_datalist.py` (writes `run_out/data/{raw,masks}/*.mrc`, `metadata.csv`, and a
`config.yaml` with `training_list`=train / `prediction_list`=test) → `run_pipeline.sh`
(partition + train, 100×300) → `predict.sh`. Per-test output:
`run_out/out/predictions/<model>/<tomo>/particle/probability_map.mrc`.
Note: DeePiCt derives its own val from `train_split`, so the JSON `validation` split isn't used
by training (add `--include-val-in-prediction` in `prepare_from_datalist.py` to also predict it).
Files: `datalist_utils.py`, `prepare_from_datalist.py`, `predict.sh`, `run_datalist.sh`.

## Resources
- **GPU** (a MIG slice like H100 `2g.20gb`/`3g.40gb` is plenty for `box_size 64`).
- **RAM**: DeePiCt loads *all* partition boxes into memory as a `TensorDataset`, then copies
  them for concat/normalize/augmentation. Budget ~2 MB × (#boxes) × 2–3. Measured: 68 tomograms
  = 11,822 boxes → **peak ~94 GB**, so use **`--mem=192G`** (an 80 GB node OOM-kills before
  epoch 0). `prepare_data.sh` also needs a real-RAM node (**≥16 GB** — a 512³ FFT per cube in
  spectrum matching; a 4 GB login/interactive shell OOM-kills it).
- **Time**: ~fixed by the schedule (100×300 ≈ 30k iterations), roughly independent of dataset
  size (full-dataset 10×300 = 22 min on one H100 MIG `3g.40gb`).

## Notes / caveats
- Padding / empty regions are handled by `min_label_fraction` box filtering — no ignore label.
- `to_device(..., gpu=None)` keeps SLURM's (MIG) `CUDA_VISIBLE_DEVICES` — do **not** pass a GPU id.
- Reported "val Dice ≈ 1 − val_loss" is inferred from the Dice loss (DeePiCt logs the loss, not a
  separate dice coefficient); "val" is the `train_split` hold-out of the training boxes, not the
  `prediction_list` tomograms. Use the prediction/postprocessing stage for held-out evaluation.
