# DeepETPicker kit

Train and score [DeepETPicker](https://github.com/cbmi-group/DeepETPicker) (a 3D ResUNet
particle picker for cryo-ET) as a detection baseline for CryoDINO, on **CZI** (Dataset440,
6 classes) and **BYU** (flagellar motor, 1 class), driven by the same detection datalists and
scored with CryoDINO's own metric code, so the numbers are directly comparable with
`3DINO/dinov2/eval/detection3d.py`.

DeepETPicker's model, loss, sampler and inference are used unchanged. The kit adds what is needed
to run it on the benchmark: converting the datalist to DeepETPicker's whole-tomogram layout,
training to CryoDINO's budget, scoring every periodic checkpoint on the full validation split, and
the patches that make its `pytorch_lightning==1.1.0`-era code run on a current stack.

## Protocol (matched to detection3d.py)

| | DeepETPicker | where |
|---|---|---|
| training data | every `training` tomogram of the datalist | `prepare_from_datalist.py` / `prepare_byu.py` |
| budget | 100 epochs x 300 iterations = 30k, batch 4, lr 5e-5 | `run_benchmark.sh` defaults |
| checkpoints | one every 5 epochs (1,500 iterations, 20 in total) | `patch_periodic_ckpt.py` |
| checkpoint choice | best score on the **full** validation split | `submit_h4h_bench_ckpt.sbatch` + `select_checkpoint.py` |
| reported number | test at the threshold(s) chosen on validation | `evaluate_czi.py` (`test_<score>_at_val_thresholds`) |
| metric | `CZIDetectionMetrics` (weighted F4) / `BYUDetectionMetrics` (F2) | `3DINO/dinov2/eval/detection_3d/metrics.py` |

lr 5e-5 is DeepETPicker's own command-line default (`options/option.py`, paired there with batch
32). At batch 4, 3e-4 and 1e-4 collapsed to one class on CZI and 3e-5 never predicted
beta-galactosidase.

Dataset-specific settings (all defaults in `run_benchmark.sh`):

| | CZI | BYU |
|---|---|---|
| block (training crop) | 72 (upstream default) | 128 (= CryoDINO's patch size) |
| label diameter | 9 | 13 |
| DeepETPicker's own val_loss | one validation tomogram | every validation tomogram (`DEP_FULL_VAL=1`) |
| `REPEAT_TRAIN` | 1 (100%), **2 (50%), 9 (10%) -- set explicitly** | `auto` |
| select job | default | `CHUNK=2`, `SKIP_CKPT_EPOCHS=4` (see `deepetpicker_byu.sh`) |

At BYU block 72, false positives sat in empty background the network never saw in training; block
128 cut the picks per split from ~3,000 to ~350. Label 13 beat 9 and 19 at 100%.

## Files

| file | what it does |
|------|--------------|
| `setup.sh` | builds the venv and patches the DeepETPicker checkout; run once |
| `run_benchmark.sh` | the pipeline, `STAGE=train` (steps 1-6) or `STAGE=eval` (7-8) on one checkpoint |
| `submit_h4h_bench_train.sbatch` | SLURM: `STAGE=train`, writes 20 periodic checkpoints |
| `submit_h4h_bench_ckpt.sbatch` | SLURM array: task *i* scores checkpoint *i* on the validation split |
| `submit_h4h_bench_select.sbatch` | SLURM: picks the best checkpoint, then scores validation + test |
| `prepare_from_datalist.py` | CZI: reassembles the training tomograms from the datalist patches -> `raw_data/` + `dataset_split.json` |
| `prepare_byu.py` | BYU: whole training tomograms cropped around the motors (patches do not tile them) |
| `make_configs.py` | writes DeepETPicker's `pre_config`; derives and validates the dataset ids |
| `patch_periodic_ckpt.py` | opt-in additions to `train.py`: periodic checkpoints, 300 iterations per epoch, repeated training set, full-validation val_loss |
| `infer_splits.py` | runs DeepETPicker's inference over the full validation/test splits in chunks (resumable) |
| `evaluate_czi.py` | scores predictions with CryoDINO's `metrics.py` (`--dataset czi\|byu`) |
| `select_checkpoint.py` | reads every checkpoint's validation score -> `checkpoint_selection.csv`, `best_f4_checkpoint.txt` |
| `datalist_utils.py` | shared datalist / patch / NIfTI / coords helpers |
| `tests/verify_data.py` | re-derives every expected value from the source patches and checks what prepare + preprocess wrote (CZI) |
| `tests/check_training_curves.py` | reads a run's TensorBoard scalars and judges whether it actually learned |
| `prepare_benchmark.py`, `score_benchmark.py` | an earlier single-job CZI flow, kept for reference; not used by the pipeline |

## Running it

Once: `bash setup.sh` (builds the venv and patches `$DEP`; `run_benchmark.sh` applies
`patch_periodic_ckpt.py` itself).

On H4H the submit scripts do the whole chain from the login node:

```bash
PERCENT=100 bash slurm_scripts/deepetpicker_czi.sh    # 100 | 50 | 10
PERCENT=100 bash slurm_scripts/deepetpicker_byu.sh
```

or by hand (`DATASET`, `DATALIST`, `OUT` are required):

```bash
E=ALL,DATASET=czi,DATALIST=<datalist.json>,OUT=<out_dir>          # + REPEAT_TRAIN=2|9 for CZI 50|10%
t=$(sbatch --parsable --export=$E submit_h4h_bench_train.sbatch)
c=$(sbatch --parsable --dependency=afterok:$t --array=0-19%4 --export=$E submit_h4h_bench_ckpt.sbatch)
sbatch --dependency=afterok:$c --export=$E submit_h4h_bench_select.sbatch
```

The result is `<out_dir>/predictions/<best checkpoint>/metrics_<czi|byu>.json`; the selection
curve is `<out_dir>/checkpoint_selection.csv`. A ckpt task that times out or lands on a broken GPU
can be resubmitted (`--array=<ids>`): inference resumes per tomogram, and the task holds and
requeues itself when its GPU is unusable.

### Pipeline steps (`run_benchmark.sh`)

| step | script | output under `<out_dir>` |
|---|---|---|
| 1 | `prepare_from_datalist.py` / `prepare_byu.py` | `raw_data/<tomo>.{mrc,coords}`, `dataset_split.json` |
| 2 | `make_configs.py` | `<DSET>_preprocess.py` (DeepETPicker's `pre_config`) |
| 3 | `bin/preprocess.py` | `coords/`, `data_std/`, `gaussian<N>/`, `data_ocp/` |
| 4 | `bin/generate_train_config.py` | `<DSET>_train.py` |
| 5 | `bin/train_bash.py` | `runs/<DSET>_train/.../checkpoints_periodic/*.ckpt`, `periodic_checkpoints.txt` |
| 6 | (checkpoint guard) | `checkpoint_used.txt` (DeepETPicker's own best-val_loss checkpoint) |
| 7 | `infer_splits.py` | `predictions/<ckpt>/<split>/<tomo>.coords` |
| 8 | `evaluate_czi.py` | `predictions/<ckpt>/metrics_<czi\|byu>.json` |

Training into an `<out_dir>` that already holds checkpoints is safe: step 5 drops a marker file
and step 6 accepts only checkpoints newer than it, so a failed training stops the run instead of
scoring a model from an earlier one.

Other knobs (environment): `EPOCHS`, `ITERS_PER_EPOCH`, `BATCH`, `LR`, `SAVE_EVERY`, `BLOCK`,
`LABEL_DIAMETER`, `CROP_MARGIN` (BYU; must be >= BLOCK/2 + BLOCK/5), `INTERNAL_VAL`,
`REPEAT_TRAIN`, `CHUNK` (tomograms per inference chunk), `LIMIT` (first N tomograms per split, for
a smoke run), `BYU_MIN_RADIUS` (leave unset: the metric's default is what CryoDINO uses), `VENV`,
`DEP`. The sbatch files find the kit through `KIT` (default `~/CryoDINO/baselines/deepetpicker_kit`),
never `$0`: `sbatch` runs a spooled copy of the script.

## Coordinate convention (the thing to get right)

A datalist patch is named `TS_a_b_<i>_<j>_<k>` and its points are `[p0, p1, p2, class, radius]` in
patch-local voxels, where `v[p0, p1, p2]` indexes the particle. So:

- assembled volume `V[p0+i, p1+j, p2+k]`, shape `(630, 630, 184)` for Dataset440;
- the MRC is written as `V.transpose(2, 1, 0)` → `(z, y, x) = (184, 630, 630)`, because `k` is the
  short axis;
- `.coords` rows are `cls+1 <tab> i <tab> j <tab> k`, because `utils/coords2labels.py` reads
  `cls, x, y, z = row` and then indexes `label_data[z, y, x]`, and
  `dataset/dataloader_DynamicLoad.py` takes the last three columns as `x, y, z` and crops
  `origin[z-s:z+s, y-s:y+s, x-s:x+s]`.
- Predictions come back in the same `(cls, x, y, z, area)` convention (`test.py` emits
  `[cls+1, coords[:,4], coords[:,3], coords[:,2]]` off the `(z, y, x)` volume), so `evaluate_czi.py`
  compares them to the GT `.coords` directly, with no re-mapping.

Classes are stored **1-based** in `.coords` (DeepETPicker uses `cls_idx - 1` to index
`ocp_diameter`); `dataset_split.json` and `metrics.json` report the original 0-based datalist ids.

## Data notes

**CZI** (`Dataset440_CZII_10440_detection_patches_128/czi_100_datalist.json`): training = 7
tomograms of 630x630x184 reassembled exactly from the 128^3 patches (overlaps agree bit-for-bit,
asserted at prepare time; 1264 particles); validation = Dataset445 (121 whole tomograms), test =
Dataset446 (364).

- 6 classes, one radius each (voxels @ 10.012 Å): `0:6.49 1:8.99 2:12.98 3:14.98 4:5.99 5:13.48`
  → `ocp_diameter = 13,19,27,31,13,27`. Diameters are rounded **up to the nearest odd** integer:
  `utils/coords2labels.py` builds a template of edge `dim` but pastes it into a window of edge
  `2*(dim//2)+1`, so an even diameter is off by one and label generation dies with *"operands could
  not be broadcast together with shapes (19,19,19) (18,18,18)"*. Upstream only ever used odd
  diameters (SHREC `7..17`, EMPIAR `23`). `make_configs.py` rejects any even diameter, including
  `LABEL_DIAMETER`.

**BYU** (`byu_detection_patches_128_motoronly/byu_100_datalist.json`): training patches do not tile
the tomograms, so `prepare_byu.py` reads the whole training tomograms, moves the patch points to
tomogram coordinates (verified by patch-vs-tomogram correlation) and keeps motors +- `CROP_MARGIN`
voxels, which covers every voxel DeepETPicker's sampler can read. Crops are cut from the
whole-tomogram-standardised volume (as at inference). One diameter is used for the class: the
median training motor radius (voxel size varies 13.1-19.7 A). Motor-free training tomograms are
skipped, since DeepETPicker samples only around particles. Validation/test: 54/55 whole tomograms.

## Patches applied to the upstream checkout

`setup.sh` edits the checkout in place (idempotent); `patch_periodic_ckpt.py` adds the opt-in
training options on top (see its docstring). DeepETPicker targets
`pytorch_lightning==1.1.0` and Python 3.7-era numpy; the cluster wheelhouse has nothing older than
PL 1.9.1 and torch 2.1.1.

1. `np.float` → **`np.float32`** and `np.int` → `int` (both removed in numpy ≥1.24). It must be
   `np.float32`, not `float`: several of these arrays go straight to `mrcfile`, which rejects
   float64 (*"dtype 'float64' cannot be converted to an MRC file mode"*). Every site sits in a
   `try/except` whose fallback was already `np.float32`, so float32 is what upstream actually gets
   on its own pinned `numpy==1.24.0`.
2. `train.py` `Trainer(...)`: `gpus=` → `devices=` + `accelerator='gpu'`, `accelerator='dp'` →
   `strategy=` (`'ddp'` only when >1 GPU; `'dp'` was removed in PL 2.0), `checkpoint_callback=` →
   moved into `callbacks=`, `profiler=True` → `profiler='simple'`.
3. `train.py`: `resume_from_checkpoint=` was removed from `Trainer` in PL 2.0 → passed as
   `fit(model, ckpt_path=...)`.
4. `train.py`: `validation_epoch_end` was **removed** in PL 2.0 and the Trainer *raises* if the
   module still defines it → renamed to `on_validation_epoch_end`, with `validation_step` outputs
   accumulated into `self.validation_step_outputs` and cleared in `on_validation_epoch_start`.
5. `test.py`: `UNetTest.load_from_checkpoint()` → explicit `load_state_dict`. The checkpoint is
   written by `UNetExperiment`, which calls `save_hyperparameters()`, so PL would forward `args`
   into `UNetTest.__init__(self)`, which takes none. Same `Trainer` fix as above.
6. `test.py`: `test_epoch_end` → `on_test_epoch_end`, same accumulator pattern as 4.
7. `utils/misc.py`: `pycm` / `scikit-plot` imports made optional (only the unreachable built-in
   scorer needs them).
8. `utils/colors.py`: `cv2` import made optional (only `plot_legend()` needs it; on Alliance
   clusters opencv is a module, not a pip wheel).
9. `train.py`: the bare `except:` around `runner.fit()` only reports through the GUI's stdout
   proxy, so a CLI failure would surface as a missing checkpoint with no traceback — it now
   prints one.
11. `train.py`: `OneCycleLR(total_steps=max_epoch, pct_start=0.1)` — the warmup phase ends at
    `pct_start*total_steps - 1`, and when that is exactly `0` the phase has zero length and
    `get_lr()` raises `ZeroDivisionError`. With `pct_start=0.1` that is **precisely
    `max_epoch == 10`**; 1 and 60 both work, so neither the smoke nor a full run would expose it.
    `pct_start` is nudged to `1.5/max_epoch` only in that degenerate case, leaving every other
    epoch count on upstream's schedule.
10. `model/residual_unet_att.py`: `out += residual` → `out = out + residual`. `out` is a ReLU
    output (from `SingleConv`), and autograd needs it intact for the ReLU backward, so on modern
    torch the first backward dies with *"one of the variables needed for gradient computation has
    been modified by an inplace operation … output 0 of Relu, is at version 1"*. All three
    `ExtResNetBlock` variants share the line.

`setup.sh` runs `git checkout -- .` in the checkout before patching, so it is deterministic rather
than layering edits on a previous run. Keep your own changes in this kit, not in `$DEP`.

PyQt5 and opencv are deliberately not installed — they are GUI-only (`MainWindow_3D_PP.py`,
`main.py`, `utils/utils.py`).

## Tests

```bash
python tests/verify_data.py --base <out_dir> --datalist <datalist.json>     # CZI, CPU, no model
python tests/check_training_curves.py <run's TensorBoard dir>               # after training
```

`verify_data.py` recomputes every expectation from the **source** `.pt` patches and the datalist,
never from the kit's own bookkeeping: random patches must round-trip out of the assembled `.mrc`
bit-for-bit, `mrc[z,y,x]` must equal `V[x,y,z]` at every labelled point, particle counts must match
an independent recount, and (after preprocess) `data_std` must be mean-0/sd-1 and the label and
occupancy volumes must carry every class. `check_training_curves.py` returns INCONCLUSIVE, not a
pass, below 5 epochs (one point per metric has no slope).

## Environment

On H4H the venv and checkout are `~/venvs/deepetpicker` + `~/deepetpicker/DeepETPicker`
(the sbatch defaults), or Reza's under `/cluster/projects/pmccaiteam/<USERNAME>/projects/cryoet/envs/`
(torch 2.7.1 cu128; used by `slurm_scripts/deepetpicker_*.sh`). Pass `DEP=` / `VENV=` to
`setup.sh` to build elsewhere.

On Alliance, `setup.sh` builds `/scratch/<USERNAME>/venvs/deepetpicker` (its default) from the wheelhouse
(`--no-index`): python 3.11, `torch==2.13.0` + `torchvision==0.28.0`,
`pytorch_lightning==2.6.5`, numpy 2.x, `batchgenerators==0.25.1`, plus pandas / mrcfile /
scikit-image / scikit-learn / scipy / matplotlib / tensorboard.

**torch 2.13.0 is not a free choice.** It is currently the only wheelhouse build with GPU kernels
for the Alliance H100s:

| wheel | `torch.cuda.get_arch_list()` |
|---|---|
| `torch 2.13.0` (CUDA 13.2) | `['sm_80','sm_90','sm_100','compute_100']` |
| `torch 2.9.1`, `2.5.1`, `2.1.2` | `[]` — dies with *"CUDA error: no kernel image is available for execution on the device"* |

That in turn forces `pytorch_lightning` 2.x, hence the hook ports in patches 4 and 6. If you change
the torch pin, re-check `get_arch_list()` before anything else — it is the cheap test that
distinguishes a usable wheel from an unusable one, and it needs no GPU.
