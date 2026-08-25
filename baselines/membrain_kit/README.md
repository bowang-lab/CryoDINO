# membrain-seg training kit

Train a **binary** 3D segmentation model with [membrain-seg](https://github.com/teamtomo/membrain-seg)
(a nnUNet-style U-Net for cryo-ET) on patch data, on a SLURM + module cluster.

It wraps membrain-seg's own model/dataloader and adds the two things you need in practice:
a **fixed iterations-per-epoch** (membrain's CLI can't set this) and **bf16 + gradient clipping**
(fp16 tends to NaN with membrain's SGD). Every sharp edge we hit is baked into the defaults.

## Files
| file | what it does |
|------|--------------|
| `convert_pt_to_nifti.py` | `.pt` patch tensors → membrain `data_dir` of `.nii.gz` (binary labels, group-based train/val split) |
| `train_membrain.py` | training entrypoint; `--smoke` for a fast GPU sanity check |
| `run_smoke.sh` | SLURM template: short GPU job that asserts the loss is finite |
| `run_train.sh` | SLURM template: the full training run |

---

## 0. Assumptions about your data
The converter expects one `.pt` (PyTorch tensor) per patch, matched by filename:

```
<src>/images/<name>.pt   # density volume, a 3D float tensor
<src>/labels/<name>.pt   # integer class ids as a tensor (0 = background, >0 = foreground)
```

Labels are **binarised** (`> 0` → foreground). Multi-class is **not** supported (membrain is
binary; its loss reserves label value `2` as an ignore region). If your data is already
`.nii.gz`/`.mrc`, skip the converter and just arrange it as in step 2 — but keep labels in
`{0,1}` and follow the pairing rule below.

---

## 1. Build the environment (once, on a login node with internet)
```bash
module load python/3.11                       # adjust to your cluster
python -m venv /path/to/venv                  # or: virtualenv --no-download /path/to/venv
source /path/to/venv/bin/activate
python -m ensurepip --upgrade                 # only if venv has no pip

# IMPORTANT: use `python -m pip`, not bare `pip` (bare pip may resolve to a different interpreter)
python -m pip install torch nibabel           # on Digital Research Alliance/CC: add --no-index
python -m pip install membrain-seg click
```
Verify: `python -m pip show membrain-seg` and `python -c "import torch;print(torch.__version__)"`.

> On a Digital Research Alliance (Compute Canada) cluster, install torch from the wheelhouse with
> `--no-index`; `membrain-seg` comes from PyPI. `click` is listed explicitly because some
> `typer` builds don't pull it.

---

## 2. Convert your data → a membrain `data_dir`
```bash
python convert_pt_to_nifti.py \
  --src /path/to/raw_patches \
  --dst /path/to/Dataset_membrain \
  --val-frac 0.2 \
  --group-regex '_patch_.*$'     # how to derive a "group" (e.g. tomogram) from a filename
```
Result:
```
Dataset_membrain/
├── imagesTr/<name>_0000.nii.gz     labelsTr/<name>.nii.gz
└── imagesVal/<name>_0000.nii.gz    labelsVal/<name>.nii.gz
```
**Pairing rule (membrain):** for label `X.nii.gz` the image must be `X_0000.nii.gz`.
The split is **by group** so whole tomograms stay on one side (no spatial leakage). Force a
specific val set with `--val-groups TE12,UF2,...`. Dry-run one patch first with `--limit 1`.

Sanity check a label has only `{0,1}`:
```bash
python -c "import nibabel,numpy,glob;f=sorted(glob.glob('/path/to/Dataset_membrain/labelsTr/*.nii.gz'))[0];print(numpy.unique(nibabel.load(f).get_fdata()))"
```

---

## 3. Smoke test (do this before every long run)
Edit the `EDIT THESE` block at the top of `run_smoke.sh` (account, `--gres`, `VENV`, `DATA_DIR`,
`KIT_DIR`), then:
```bash
sbatch run_smoke.sh
```
It must finish with **`SMOKE TEST PASSED: train_loss is FINITE`**. If it prints
`SMOKE TEST FAILED … NaN`, fix the data before wasting GPU hours (see Troubleshooting).

---

## 4. Full training run
Edit the `EDIT THESE` block in `run_train.sh`, then:
```bash
sbatch run_train.sh
squeue -u $USER
```
Defaults: 100 epochs × 300 iterations, batch 2, bf16, deep supervision + full augmentation.
Outputs go to `$WORK_DIR/checkpoints/` and `$WORK_DIR/logs/`.

Watch progress (finite & falling `train_loss`, rising `val_dice`):
```bash
python - <<'EOF'
import csv,glob
m=sorted(glob.glob('/path/to/workdir/logs/lightning_logs/version_*/metrics.csv'))[-1]
for r in csv.DictReader(open(m)):
    if r.get('val_dice'): print(r['epoch'], 'train_loss',r['train_loss'],'val_dice',r['val_dice'])
EOF
```

**Copy the best checkpoint off scratch** (scratch is often purged):
```bash
cp "$WORK_DIR"/checkpoints/membrain_binary-*val_loss*.ckpt /durable/storage/
```

---

## 5. Inference (after training)
```bash
membrain segment --tomogram-path tomo.mrc \
  --ckpt-path /durable/storage/membrain_binary-<best>.ckpt \
  --out-folder predictions/
```

---

## Key knobs (`train_membrain.py --help`)
- `--max-epochs`, `--iters-per-epoch` — training length (an "epoch" here = this many random crops).
- `--batch-size`, `--num-workers`.
- `--precision` — keep `bf16-mixed` on A100/H100; `16-mixed` (fp16) can NaN.
- `--no-aug`, `--no-deep-supervision` — turn off for faster (weaker) runs.

## Troubleshooting
- **`train_loss` is NaN from epoch 0** — almost always a label problem: a random crop with **no
  valid voxels** (e.g. entirely membrain's ignore label `2`) makes the Dice/CE loss divide by
  zero. Keep labels `{0,1}` and never mark large padding regions as `2`. This is why the smoke
  test asserts finiteness — a NaN loss does **not** crash; it silently trains to garbage.
- **`val_dice` stuck at 0 for many epochs** — expected for a while when foreground is very sparse
  (~1% of voxels): the model first learns "all background". If it never lifts, reduce
  all-background crops or add class weighting / a Tversky/focal loss (needs a small membrain edit).
- **OOM / job killed** — training loads full volumes on the fly; lower `--num-workers`, raise
  `--mem`, or pre-crop to smaller patches. Do not disable on-the-fly loading for large datasets.
- **`No supported gpu backend found` / `CUDA unknown error`** — usually a flaky node; resubmit
  (optionally `sbatch --exclude=<bad_node> ...`).
- **MIG slices + long walltime** — on some clusters MIG GPUs live only in short-walltime
  partitions; check `sinfo -o "%P %l %G"` and request a partition that offers your slice for 24h.

## One-shot from a datalist JSON (train → infer on test → save per sample)
Given a MONAI-style datalist `{"training":[…],"validation":[…],"test":[…]}` of
`{"image","label"}` entries (paths used **verbatim**; `.pt` cubes or `.nii.gz`/`.mrc` auto-detected,
labels binarised `>0`):
```bash
EPOCHS=100 ITERS=300 VENV=/path/to/venv bash run_datalist.sh datalist.json /path/to/run_out
```
Runs `prepare_from_datalist.py` (builds `run_out/data_dir` + `test_list.json`) →
`train_membrain.py` → picks the best checkpoint → `infer_and_save.py`. Per-test outputs land in
`run_out/test_predictions/<name>/` (`*_segmented.mrc` + `*_scores.mrc`), with Dice in
`test_predictions/metrics.csv` when labels are present. (membrain reads MRC only, so `.nii.gz`
test volumes are auto-converted to `.mrc`.) Dry-run the JSON with
`python prepare_from_datalist.py --datalist … --data-dir /tmp/x --test-out /tmp/x.json --dry-run`.
Files: `datalist_utils.py`, `prepare_from_datalist.py`, `infer_and_save.py`, `run_datalist.sh`.
