# DeepETPicker kit

Train and evaluate [DeepETPicker](https://github.com/cbmi-group/DeepETPicker) (a 3D ResUNet
particle picker for cryo-ET) as a benchmark baseline, driven by a **detection** datalist JSON
(point labels), on a SLURM + module cluster.

Sibling of `membrain_kit/` and `deepict/`, and the first of the three to handle point labels
rather than segmentation masks. It adds the two things you need in practice: a **converter from
128³ labelled patches to the whole-tomogram `.mrc` + `.coords` layout DeepETPicker requires**, and
a **scorer**, because DeepETPicker's own multi-class scorer is unreachable from its CLI. Ten
patches to the upstream checkout (§ *Patches*) are what make its `pytorch_lightning==1.1.0`-era
code run on a current stack; every sharp edge we hit is baked in.

## Files
| file | what it does |
|------|--------------|
| `setup.sh` | builds the venv and patches the DeepETPicker checkout; run once |
| `prepare_from_datalist.py` | datalist patches → `raw_data/<tomo>.{mrc,coords}` + `dataset_split.json` |
| `make_configs.py` | writes DeepETPicker's `pre_config`; derives and validates the train/val/test ids |
| `evaluate.py` | CZII-style detection metrics → `metrics.json` |
| `datalist_utils.py` | shared datalist/patch/coords helpers |
| `run_datalist.sh` | the whole pipeline, prepare → preprocess → train → infer → score |
| `smoke.sbatch` | SLURM: 1-epoch end-to-end run, to prove the plumbing before a real one |
| `submit.sbatch` | SLURM: the full run |
| `tests/verify_data.py` | re-derives every expected value from the source patches and checks what prepare + preprocess wrote |
| `tests/test_evaluate.py` | known-answer test of `evaluate.py` (no GPU, no trained model) |
| `tests/check_training_curves.py` | reads a run's TensorBoard scalars and judges whether it actually learned |
| `curves.sbatch` | SLURM: short run (default 10 epochs) just to produce learning curves |

## Quickstart
```bash
bash setup.sh                                   # once: venv + source patches
sbatch smoke.sbatch                             # 1 epoch end-to-end (~4 min once it runs)
sbatch submit.sbatch                            # the full run
```

## What it does

DeepETPicker works on **whole tomograms**: a `raw_data/` folder holding `<tomo>.mrc` next to
`<tomo>.coords`. The datalist instead ships 128³ patches with point labels, so
`prepare_from_datalist.py` reassembles each tomogram from its patches and merges the points.

For Dataset440 that reassembly is **exact**: the patches are crops of one globally normalised
volume, so overlapping patches agree bit-for-bit (asserted at prepare time, along with full
coverage of the assembled volume and the expected particle counts).

| step | script | output under `<out_dir>` |
|---|---|---|
| 1 | `prepare_from_datalist.py` | `raw_data/<tomo>.{mrc,coords}`, `dataset_split.json` |
| 2 | `make_configs.py` | `<DSET>_preprocess.py` (DeepETPicker's `pre_config`) |
| 3 | `bin/preprocess.py` | `coords/`, `data_std/`, `gaussian<N>/`, `data_ocp/` |
| 4 | `bin/generate_train_config.py` | `<DSET>_train.py` |
| 5 | `bin/train_bash.py` | `runs/<DSET>_train/.../version_N/checkpoints/*.ckpt` |
| 6 | `bin/test_bash.py` | `.../version_N/PredictedLabels/Coords_{All,withArea}/<tomo>.coords` |
| 7 | `evaluate.py` | `.../PredictedLabels/metrics.json` |

`<out_dir>` becomes DeepETPicker's `base_path`. Put it on `/scratch` — it holds ~2 GB of tomograms
plus the same again in normalised copies and label volumes — and copy the small artefacts
(`PredictedLabels/`, `metrics.json`, the checkpoint) somewhere durable afterwards, since scratch is
purged.

Step 5 drops a marker file before training and step 6 accepts only a checkpoint newer than it. Re-running
into an `<out_dir>` that already holds checkpoints is therefore safe: if training fails, the run
stops instead of silently scoring a model from an earlier run.

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
  `[cls+1, coords[:,4], coords[:,3], coords[:,2]]` off the `(z, y, x)` volume), so `evaluate.py`
  compares them to the GT `.coords` directly, with no re-mapping.

Classes are stored **1-based** in `.coords` (DeepETPicker uses `cls_idx - 1` to index
`ocp_diameter`); `dataset_split.json` and `metrics.json` report the original 0-based datalist ids.

## Splits

DeepETPicker takes `--train_set_ids`/`--val_set_ids` as indices into the **sorted** tomogram list,
which is exactly the order `prepare_from_datalist.py` records in `dataset_split.json`. `make_configs.py`
defaults to *last = test, second-to-last = val, the rest = train* and refuses a choice that would
leave the train ids non-contiguous — `bin/train_bash.py` derives `data_split` from
`len(train_list)`, not from the ids, so train must be `0..n-1`. Override with `--val-id` / `--test-id`.

Note `0-4` in `--train_set_ids` expands **inclusive** (`np.arange(lo, hi+1)`), contradicting the
upstream tutorial text.

## Scoring

DeepETPicker's own scorer (`utils/misc.py:cal_metrics_MultiCls`) is unreachable from `bin/`: it is
gated on `'test' in args.test_mode`, but `train_bash.py` runs with `test_mode='val'` and
`test_bash.py` with `'test_only'`. So this kit scores with `evaluate.py`, which follows the CZII
convention — a prediction matches a same-class ground-truth particle within `0.5 × radius`, one
prediction per particle, greedy by distance — and reports per-class and macro/micro precision,
recall, F1 and F-beta (β=4). The JSON is shaped like `baselines_metrics/{deepict,membrain}/metrics.json`
(`per_image` keyed by tomogram).

## Patches applied to the upstream checkout

`setup.sh` edits the checkout in place (idempotent). DeepETPicker targets
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

Both run on CPU in a couple of minutes and need no trained model:

```bash
python tests/verify_data.py --base <out_dir> --datalist <datalist.json>
python tests/test_evaluate.py --base <out_dir>
```

`verify_data.py` recomputes every expectation from the **source** `.pt` patches and the datalist —
never from the kit's own bookkeeping — so agreement means the pipeline wrote what it should have:
random patches must round-trip out of the assembled `.mrc` bit-for-bit, `mrc[z,y,x]` must equal
`V[x,y,z]` at every labelled point, particles must be dark relative to the volume mean, particle
counts must match an independent recount, and (if preprocess has run) `data_std` must be
mean-0/sd-1, the label volume must carry every class, and `data_ocp`'s max must equal the particle
count exactly.

`test_evaluate.py` synthesises a prediction file from real ground truth with a known number of
hits, misses and false positives, and checks `evaluate.py` recovers exactly those counts and the
closed-form precision/recall/F-beta — plus that a pick just outside `0.5 × radius` does **not**
match.

`check_training_curves.py` (needs a finished run, no GPU) answers "did it actually learn?" from the
TensorBoard scalars — `train_loss`/`val_loss` must fall, `val_f1`/`val_precision`/`val_recall`/
`val_iou` must rise, and it flags a `val_loss` that bottoms out early. Crucially it returns
**INCONCLUSIVE**, not a pass, below 5 epochs: a 1-epoch run logs *one point per metric*, and a
single point has no slope. Do not wire it into `run_datalist.sh` as a hard gate — see the
`val_precision` note under Status.

## Environment

`setup.sh` builds `/scratch/cyyu/venvs/deepetpicker` from the Alliance wheelhouse
(`--no-index`): python 3.11, `torch==2.13.0` + `torchvision==0.28.0`,
`pytorch_lightning==2.6.5`, numpy 2.x, `batchgenerators==0.25.1`, plus pandas / mrcfile /
scikit-image / scikit-learn / scipy / matplotlib / tensorboard.

**torch 2.13.0 is not a free choice.** It is currently the only wheelhouse build with GPU kernels
for this cluster's H100s:

| wheel | `torch.cuda.get_arch_list()` |
|---|---|
| `torch 2.13.0` (CUDA 13.2) | `['sm_80','sm_90','sm_100','compute_100']` |
| `torch 2.9.1`, `2.5.1`, `2.1.2` | `[]` — dies with *"CUDA error: no kernel image is available for execution on the device"* |

That in turn forces `pytorch_lightning` 2.x, hence the hook ports in patches 4 and 6. If you change
the torch pin, re-check `get_arch_list()` before anything else — it is the cheap test that
distinguishes a usable wheel from an unusable one, and it needs no GPU.

## Knobs

`smoke.sbatch` and `submit.sbatch` locate the kit through `KIT` (default: this directory's
absolute path). They must not derive it from `$0` — `sbatch` copies the batch script to a per-job
spool dir on the compute node, so `$0` points at `/localscratch/spool/slurmd/job<ID>/` and the
script is not found there. If you move the kit, set `KIT=<kit dir>` or edit the default.

`run_datalist.sh` reads these from the environment:
`EPOCHS` (60), `BATCH` (8), `BLOCK` (72, the training patch size — must be a multiple of 8),
`PAD` (12), `LR` (0.001), `THRESH` (0.5), `GPU` (0), `LABEL_DIAMETER` (9), `DSET` (`Dataset440`),
`VENV`, `DEP`.

Re-running `run_datalist.sh` on an existing `<out_dir>` reuses the tomograms already in
`raw_data/` — reassembly reads ~9 GB of patches and takes ~20 min. Pass `--force` to
`prepare_from_datalist.py` to rebuild them.

`LABEL_DIAMETER` is the weak gaussian label; it must be smaller than the smallest real particle
diameter (`make_configs.py` enforces this). The real per-class diameters go to DeepETPicker as
`ocp_diameter`, derived from the radii in the datalist.

## Dataset440 specifics

`Dataset440_CZII_10440_detection_patches_128/czi_100_datalist.json`:

- Only the **`training`** split is usable. `validation` (121) and `test` (364) point at
  `Dataset445`/`Dataset446` whole tomograms that are not on this filesystem, and all their points
  are the `-100` "no annotation" sentinel. `prepare_from_datalist.py` reports and skips any entry
  that is not a 128³ patch.
- The datalist's `image` paths are from the machine it was built on; the patches are resolved by
  basename under `--image-root` (default `<datalist dir>/images`).
- 7 tomograms, 630×630×184 each, **1264 unique particles**
  (TS_5_4 140, TS_69_2 143, TS_6_4 193, TS_6_6 143, TS_73_6 216, TS_86_3 224, TS_99_9 205) —
  asserted at prepare time.
- 6 classes, one radius each (voxels @ 10.012 Å): `0:6.49 1:8.99 2:12.98 3:14.98 4:5.99 5:13.48`
  → `ocp_diameter = 13,19,27,31,13,27`. Diameters are rounded **up to the nearest odd** integer:
  `utils/coords2labels.py` builds a template of edge `dim` but pastes it into a window of edge
  `2*(dim//2)+1`, so an even diameter is off by one and label generation dies with *"operands could
  not be broadcast together with shapes (19,19,19) (18,18,18)"*. Upstream only ever used odd
  diameters (SHREC `7..17`, EMPIAR `23`). `make_configs.py` rejects any even diameter, including
  `LABEL_DIAMETER`.
- Default split: train `TS_5_4, TS_69_2, TS_6_4, TS_6_6, TS_73_6`; val `TS_86_3`; test `TS_99_9`.

With ~900 particle-centred training samples per epoch at `batch_size 8` this is a small run; the
data is thin for the two rare classes (9–22 particles per tomogram), so their per-class numbers are
noisy by construction. Leave-one-out over the 7 tomograms is the obvious follow-up — same kit,
7 runs with different `--val-id`/`--test-id`.

## Status

Verified to the level the project accepted: **plumbing + 10 epochs of learning**. A 60-epoch
production run was deliberately not done.

*Plumbing* (`smoke.sbatch`, `EPOCHS=1`, exit 0): all seven stages ran, all seven tomograms picked,
`metrics.json` written. `tests/verify_data.py` and `tests/test_evaluate.py` pass on it. Note a
1-epoch run logs **one point per metric**, so it says nothing about whether the model learns —
`tests/check_training_curves.py` correctly calls it `INCONCLUSIVE`.

*Learning* (`curves.sbatch`, `EPOCHS=10`, job 58904233, exit 0):

| metric | first | last | delta | |
|---|---|---|---|---|
| `train_loss` | 0.9108 | 0.5351 | −0.3758 | falling |
| `val_loss` | 0.8927 | 0.6186 | −0.2741 | falling |
| `val_f1` | 0.1798 | 0.3436 | +0.1638 | rising |
| `val_recall` | 0.3959 | 0.6123 | +0.2164 | rising |
| `val_iou` | 0.1935 | 0.4168 | +0.2233 | rising |
| `val_precision` | 0.5096 | 0.4425 | −0.0671 | **falling** |

Losses fall smoothly and track each other — no overfitting. Class collapse is gone, which is the
clearest evidence training works:

| | test `TS_99_9` micro F1 | classes predicted |
|---|---|---|
| 1 epoch | 0.076 | 1 of 6 (only ribosome) |
| 10 epochs | **0.487** | **6 of 6** |

**`val_precision` falls**, which `check_training_curves.py` flags as `CURVES PROBLEM`. Read against
recall it is the model trading precision for recall as it stops being timid — recall gains +0.22 for
−0.07 of precision, and the end-to-end scores agree (452 picks against 205 real particles: micro
P=0.354 / R=0.780). That reading is an **interpretation, not a proven fact** — a longer run would
confirm or refute it. Consequently, **do not** wire the checker into `run_datalist.sh` as a hard
gate; it would fail healthy runs.

### Known gaps

- **`val_loss` was still falling at epoch 8 of 9** — 10 epochs is undertrained. The numbers above
  are a verification result, *not* a benchmark result to put beside deepict/membrain.
- **`submit.sbatch` has never actually executed.** Every other script has. It was submitted twice
  and cancelled both times. Since bug 4 was an sbatch-wrapper bug visible only on real submission,
  treat the first real `sbatch submit.sbatch` as unproven. `smoke.sbatch` and `curves.sbatch`, which
  are near-identical, both work.
- **Untested paths:** `setup.sh`'s `git clone` branch (the checkout always existed), the multi-GPU
  `strategy='ddp'` branch (always one GPU), `prepare_from_datalist.py --force`, and
  `make_configs.py`'s error paths.
- **Only ever run on Dataset440.** The kit is written to be datalist-driven, but that generality is
  a design claim, not a tested one — and `prepare_from_datalist.py` still carries a hardcoded
  `EXPECTED_PARTICLES` table for Dataset440 (disable with `--no-expect-check`).
- **`metrics.json` is not directly comparable to `baselines_metrics/{deepict,membrain}`.** It shares
  the `per_image` structure, but those hold *segmentation* metrics (dice, hd95) while this holds
  *detection* metrics (precision/recall/F-beta). Correct for the task; not a drop-in comparison.

### Results

Copied off purge-eligible scratch to
`/home/cyyu/projects/def-wanglab-ab/cyyu/deepetpicker_ds440_results/` (17 MB): `PredictedLabels/`
(coords for all 7 tomograms + `metrics.json`), the best checkpoint `epoch=8-step=945.ckpt`
(`ModelCheckpoint` kept the best-`val_loss` epoch, not the last), the TensorBoard events, the train
config, `dataset_split.json` and the run log. The 7.7 GB of reassembled tomograms stays on scratch —
`prepare_from_datalist.py` rebuilds them in ~20 min.

Not `cryodino/baselines_out/`, which is owned by another user and group-read-only.

### Bugs found and fixed

| # | bug | where |
|---|---|---|
| 1 | `np.float` → `float` made MRC buffers float64, which `mrcfile` rejects | this kit's patch 1 |
| 2 | even `ocp_diameter` values break `coords2labels`'s template broadcast | upstream |
| 3 | wheelhouse torch < 2.13 ships no H100 kernels (`get_arch_list() == []`) | environment |
| 4 | `$0`/`BASH_SOURCE` in an sbatch script resolves to the SLURM spool dir | this kit's wrappers |
| 5 | `out += residual` mutates a ReLU output and breaks autograd | upstream |
| 6 | `OneCycleLR` divides by zero at **exactly** `max_epoch == 10` | upstream |

Bug 6 is the argument for the step-6 checkpoint guard: when training died, the run stopped with
"training wrote no checkpoint" instead of silently scoring the leftover 1-epoch model and returning
a well-formed `metrics.json` that looked like a 10-epoch result.
