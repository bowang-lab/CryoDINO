# CryoDINO Manuscript — Experiment Plan

Mirrors the figure structure of the parent 3DINO paper (`manuscript/s41746-025-02035-w (2).pdf`,
*"A generalizable 3D framework and model for self-supervised learning in medical imaging"*),
adapted to CryoDINO's segmentation and detection datasets.

---

## 1. Fig. 1c analog — PCA / MHSA visualization, B200 backbone

## Daphne 
Original image / PCA of patch-level features / MHSA attention map, per representative tomogram.
Starting from scratch (previous renders were H100-only). Run with **both** B200 checkpoints per
image and compare which gives the better/sharper visualization — no need to pin down exactly
which dataset each image belongs to, just reuse what's already staged.

**Images** (already staged, reused from the existing H100-era `attention_maps/` set):

## Sumin add the path of images on h4h and add 3 more samples
| Image | Path |
|---|---|
| Sample 1 | `visualization/attention_maps/imgs/TS_0001_0000.nii.gz` |
| Sample 2 | `visualization/attention_maps/imgs/TS_0009_0000.nii.gz` |
| Sample 3 | `visualization/attention_maps/imgs/TS_027_0000.nii.gz` |

**Checkpoints to compare:**

| Checkpoint | Path | Config | `--image-size` |
|---|---|---|---|
| B200 highres128 (best overall fine-tuning result) | `/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/ssl3d_run_b200_high_res_128/eval/training_6249/teacher_checkpoint.pth` | `3DINO/dinov2/configs/train/vit3d_highres.yaml` | 128 |
| B200 highres112 (LP-sweep robust winner) | `/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/ssl3d_run_b200_high_res/eval/training_6249/teacher_checkpoint.pth` | `3DINO/dinov2/configs/train/vit3d_highres_112.yaml` | 112 |

**Samples** (confirmed via `slurm_scripts/vis_attention_maps.sh`'s own comments — all 3 are actually
**Dataset001/CZII-10001**, not one-per-dataset as first assumed):
- `TS_0001` → training sample
- `TS_0009` → internal val/test sample
- `TS_027` → external val/test sample (likely Dataset000, the unseen-test set from `results.md`)

**Scripts — two-stage pipeline, both already exist:**

1. `slurm_scripts/vis_attention_maps.sh` — orchestrates `3DINO/dinov2/eval/vis_pca_cryoet.py` per
   sample × vis-type (`pca`/`mhsa`), with `--input-type sliding_window` (not the script's own
   default `full_image` — important to keep). Currently hardcodes the **H100** checkpoint
   (`ssl3d_run_h100_high_res/eval/training_9374`) and `IMAGE_SIZE=112` — needs editing in place for
   each B200 checkpoint we're comparing (§1 above), run once per checkpoint. Also note: the
   existing `vis_output/TS_0001/` dirs on disk have both plain `pca`/`mhsa` AND `pca_448`/`mhsa_448`
   variants — the `_448` ones are a second, higher-resolution pass (likely `--image-size 448`) that
   `plot_all_heads.py` specifically expects as input; the current script only produces one
   resolution per run, so generating both means running it twice (112-ish sliding-window pass +
   a 448 pass) per checkpoint.
2. `visualization/plot_all_heads.py` — takes a sample's `pca_448`/`mhsa_448` output dirs (+ optional
   label NIfTI) and composes the final `{name}_best_planes.png` (4-row grid: 3 depth slices + best
   foreground slice, columns = original/label/PCA-RGB/MHSA-heads). This is the actual Fig. 1c-style
   panel, not `vis_pca_cryoet.py`'s raw per-slice output directly.

**Commands** (edit `WEIGHTS`/`CONFIG`/`IMAGE_SIZE` in `vis_attention_maps.sh` in place per
checkpoint, resubmit once per checkpoint, then run `plot_all_heads.py` per sample):

```bash
# 1) edit slurm_scripts/vis_attention_maps.sh:
#    WEIGHTS -> B200 checkpoint path, CONFIG -> matching config, IMAGE_SIZE -> 128 or 112
#    (plus a second edit+resubmit at --image-size 448 for the pca_448/mhsa_448 inputs plot_all_heads.py needs)
sbatch slurm_scripts/vis_attention_maps.sh

# 2) compose the final panel per sample, once vis_output/<name>/{pca_448,mhsa_448} exist:
python visualization/plot_all_heads.py \
    --pca-dir    visualization/attention_maps/vis_output/TS_0001/pca_448 \
    --mhsa-dir   visualization/attention_maps/vis_output/TS_0001/mhsa_448 \
    --output-dir visualization/attention_maps \
    --name       TS_0001 \
    --lbl-path   dataset/downstream/Dataset001_CZII_10001/labelsTr/TS_0001.nii.gz
```

Status: ❌ not yet run with B200. Once both checkpoints' outputs exist, compare visually per sample
and pick the better one (or keep both as a supplementary comparison). Note: we could not determine
the provenance of the existing `TS_0001_all_heads.png` (not produced by the current
`plot_all_heads.py`, which only ever writes `_best_planes.png`) — treat it as a stale/orphaned
artifact rather than something to reproduce.

---

## 2. Segmentation benchmark

**Datasets:** Dataset001 (CZII-10001, 4-class: bg/cytoplasm/organelles/membrane), Dataset010
(CZII-10010, binary membrane), EMPIAR-10989 (binary actin filaments), EMPIAR-12049/Dataset049
(4-class after remap: bg/membranes/microtubules/ribosomes — raw data is 6-class
matrix/membranes/microtubules/ribosomes/centrioles, with matrix+centrioles merged into background;
see `augmentations.py`'s `_remap_12049`).

**Methods:** CryoDINO B200 (two backbone checkpoints compared — `highres128`/`training_6249` and
`highres112`/`training_6249`), CryoDINO random-init, UNet (scratch), UNETR (scratch), DeePiCt,
membrain-seg.

### 2.1 100% data — ✅ CONFIRMED (test Dice, per-class in brackets)

**Dataset001** (per-class order: [bg, cytoplasm, organelles, membrane]):

| Method | Test Dice | Per-class |
|---|---|---|
| CryoDINO (highres128) | 0.8912 | [0.9926, 0.9547, 0.8918, 0.7256] |
| CryoDINO (highres112) | **0.8918** | [0.9930, 0.9575, 0.8957, 0.7211] |
| CryoDINO random-init | 0.8530 | [0.9904, 0.9380, 0.7898, 0.6937] |
| UNet (scratch) | 0.8453 | [0.9857, 0.9380, 0.7844, 0.6731] |
| UNETR (scratch) | 0.8272 | [0.9891, 0.9088, 0.7477, 0.6632] |
| DeePiCt | ❌ not run | |
| membrain-seg | ❌ not run | |

**Dataset010** (per-class order: [bg, membrane]):

| Method | Test Dice | Per-class |
|---|---|---|
| CryoDINO (highres128) | **0.8474** | [0.9977, 0.6972] |
| CryoDINO (highres112) | 0.8289 | [0.9974, 0.6603] |
| CryoDINO random-init | 0.8113 | [0.9969, 0.6257] |
| UNet (scratch) | 0.7971 | [0.9968, 0.5974] |
| UNETR (scratch) | 0.8191 | [0.9974, 0.6409] |
| DeePiCt | 0.7291 | [0.9909, 0.4672] |
| membrain-seg | 0.7459 | [0.9953, 0.4966] |

**EMPIAR-10989** (per-class order: [bg, actin filament]):

| Method | Test Dice | Per-class |
|---|---|---|
| CryoDINO (highres128) | **0.6758** | [0.9974, 0.3542] |
| CryoDINO (highres112) | 0.6555 | [0.9963, 0.3147] |
| CryoDINO random-init | 0.6659 | [0.9975, 0.3342] |
| UNet (scratch) | 0.6267 | [0.9975, 0.2559] |
| UNETR (scratch) | 0.6490 | [0.9969, 0.3012] |
| DeePiCt | ❌ not run | |
| membrain-seg | ❌ not run | |

**Dataset049/EMPIAR-12049** (per-class order: [bg, membranes, microtubules, ribosomes], 4-class
after remap):

| Method | Test Dice | Per-class |
|---|---|---|
| CryoDINO (highres128) | 0.4867 | [0.9877, 0.3318, 0.3796, 0.2476] |
| CryoDINO (highres112) | **0.4889** | [0.9880, 0.3327, 0.3893, 0.2454] |
| CryoDINO random-init | 0.4877 | [0.9875, 0.3449, 0.3848, 0.2335] |
| UNet (scratch) | 0.4736 | [0.9862, 0.3184, 0.3609, 0.2290] |
| UNETR (scratch) | 0.4132 | [0.9870, 0.2548, 0.2095, 0.2015] |
| DeePiCt | ❌ not run | |
| membrain-seg | ❌ not run | |

**Observations:**
- `highres128` wins on DS010 and EMPIAR-10989; `highres112` wins on DS001 and Dataset049 — no
  single backbone dominates, consistent with the earlier LP-sweep finding that preference is
  dataset-dependent. ❓ **Decide:** report both per dataset, or pick one backbone as "the" CryoDINO
  number across all four for a cleaner figure?
- CryoDINO random-init re-run on all 4 datasets is now ✅ done and valid (the earlier
  `train_3dino_ft_random_init_patches.sh` config/6-class bugs are fixed; two attempts hit host-RAM
  OOM mid-training — see §2.4 note — resolved by bumping `--mem` to 800G and rerunning just the
  affected datasets). Pretraining's benefit is clearly dataset-dependent: it's large on Dataset001
  (+0.039) and Dataset010 (+0.036), smaller on EMPIAR-10989 (+0.010), and essentially **zero** on
  Dataset049 (+0.001) — on the hardest, most class-imbalanced dataset, the pretrained backbone
  isn't meaningfully helping over random init.

**Source paths** (under `/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/`):

| Method | Path |
|---|---|
| CryoDINO highres128 | `finetuning/ssl3d_run_b200_high_res_128_training_6249_Dataset{001_CZII_10001,010_CZII_10010,989_EMPIAR_10989_transposed,049_EMPIAR_12049_transposed}_patches512_vit_adapter/results.json` |
| CryoDINO highres112 | `finetuning/ssl3d_run_b200_high_res_training_6249_Dataset{...}_patches512_vit_adapter/results.json` |
| CryoDINO random-init | `finetuning/random_init_Dataset{...}_patches512_vit_adapter/results.json` |
| UNet / UNETR | `baselines/{unet,unetr}_Dataset{...}_patches512/results.json` |
| DeePiCt (DS010 only so far) | `baselines_out/deepict_ds010/cryodino_metrics/metrics.json` |
| membrain-seg (DS010 only so far) | `baselines_out/membrain_ds010/cryodino_metrics/metrics.json` |

### 2.2 10%/50% data — ✅ CONFIRMED (CryoDINO best-checkpoint only, test Dice)

Each dataset's best backbone from §2.1 (highres112 for Dataset001/Dataset049, highres128 for
Dataset010/EMPIAR-10989), run via `slurm_scripts/train_3dino_ft_b200_best_ckpt_patches_{10,50}pct.sh`.
Both runs completed cleanly (no errors in either log); val/test counts matched the 100% file with
only `training` shrinking, confirming the percent-datalist files were read correctly.

| Dataset | Backbone | 10% | 50% | 100% (reference) |
|---|---|---|---|---|
| Dataset001/CZII-10001 | highres112 | 0.8630 | 0.8891 | 0.8918 |
| Dataset010/CZII-10010 | highres128 | 0.8261 | 0.8362 | 0.8474 |
| EMPIAR-10989 | highres128 | 0.6747 | 0.6626 | 0.6758 |
| Dataset049/EMPIAR-12049 | highres112 | 0.3428 | 0.4915 | 0.4889 |

Source paths: `finetuning/ssl3d_run_b200_high_res{,_128}_training_6249_Dataset{...}_patches512_{10,50}percent_vit_adapter/results.json`
(backbone tag matches the per-dataset mapping above, same convention as the 100% runs in §2.1).

**CryoDINO random-init, 10%/50%** (`slurm_scripts/train_3dino_ft_random_init_patches_{10,50}pct.sh`,
`--mem=800G`): 50% ✅ done, all 4 datasets completed cleanly (no OOM); 10% 🔄 still running.

| Dataset | 10% | 50% | 100% (reference) |
|---|---|---|---|
| Dataset001/CZII-10001 | 🔄 running | 0.8278 | 0.8530 |
| Dataset010/CZII-10010 | 🔄 running | 0.8055 | 0.8113 |
| EMPIAR-10989 | 🔄 running | 0.6920 | 0.6659 |
| Dataset049/EMPIAR-12049 | 🔄 running | 0.4810 | 0.4877 |

Source paths: `finetuning/random_init_Dataset{...}_patches512_{10,50}percent_vit_adapter/results.json`.

Still needed at 10%/50%: UNet/UNETR (running), DeePiCt/membrain-seg (not started) — see §2.4.

### 2.3 Datalists (10%/50%) 

Unlike the detection rows in §3.1, these segmentation rows in the CSV are marked `verified=no` —
the paths exist on disk (confirmed earlier via `ls experiments/` on H4H) but the `train_patches`/
`train_tomograms`/`val`/`test` counts haven't actually been checked/filled in yet.

| Dataset | % | Datalist path | `--dataset-name` | `--base-data-dir` |
|---|---|---|---|---|
| Dataset001 | 100 | `experiments/Dataset001_CZII_10001_patches512_100_datalist.json` | `Dataset001_CZII_10001_patches512` | `experiments` |
| Dataset001 | 50 | `experiments/Dataset001_CZII_10001_patches512_50percent_100_datalist.json` | `Dataset001_CZII_10001_patches512_50percent` | `experiments` |
| Dataset001 | 10 | `experiments/Dataset001_CZII_10001_patches512_10percent_100_datalist.json` | `Dataset001_CZII_10001_patches512_10percent` | `experiments` |
| Dataset010 | 100 | `experiments/Dataset010_CZII_10010_patches512_100_datalist.json` | `Dataset010_CZII_10010_patches512` | `experiments` |
| Dataset010 | 50 | `experiments/Dataset010_CZII_10010_patches512_50percent_100_datalist.json` | `Dataset010_CZII_10010_patches512_50percent` | `experiments` |
| Dataset010 | 10 | `experiments/Dataset010_CZII_10010_patches512_10percent_100_datalist.json` | `Dataset010_CZII_10010_patches512_10percent` | `experiments` |
| EMPIAR-10989 | 100 | `experiments/Dataset989_EMPIAR_10989_transposed_patches512_100_datalist.json` | `Dataset989_EMPIAR_10989_transposed_patches512` | `experiments` |
| EMPIAR-10989 | 50 | `experiments/Dataset989_EMPIAR_10989_transposed_patches512_50percent_100_datalist.json` | `Dataset989_EMPIAR_10989_transposed_patches512_50percent` | `experiments` |
| EMPIAR-10989 | 10 | `experiments/Dataset989_EMPIAR_10989_transposed_patches512_10percent_100_datalist.json` | `Dataset989_EMPIAR_10989_transposed_patches512_10percent` | `experiments` |
| Dataset049 | 100 | `experiments/Dataset049_EMPIAR_12049_transposed_patches512_100_datalist.json` | `Dataset049_EMPIAR_12049_transposed_patches512` | `experiments` |
| Dataset049 | 50 | `experiments/Dataset049_EMPIAR_12049_transposed_patches512_50percent_100_datalist.json` | `Dataset049_EMPIAR_12049_transposed_patches512_50percent` | `experiments` |
| Dataset049 | 10 | `experiments/Dataset049_EMPIAR_12049_transposed_patches512_10percent_100_datalist.json` | `Dataset049_EMPIAR_12049_transposed_patches512_10percent` | `experiments` |

(`experiments` = `/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments`. Note the
mechanism differs from detection's: here the **filename** carries the percent suffix and
`--base-data-dir` stays constant, whereas detection puts the percent in a **subdirectory** and
keeps the filename constant — both land on the same `{base-data-dir}/{dataset-name}_100_datalist.json`
lookup pattern `loaders.py` expects.)

### 2.4 Still needed

- ✅ **CryoDINO random-init, 100%, all 4 datasets — done.** Hit host-RAM OOM twice along the way
  (`DataLoader worker killed by signal`, cause: host-RAM pressure accumulating across the 4
  sequential per-dataset training calls within one job/cgroup — each a fresh Python subprocess, but
  sharing the same `--mem` ceiling) — fixed by bumping `slurm_scripts/train_3dino_ft_random_init_patches.sh`'s
  `--mem` 500G→700G→800G and rerunning just the datasets that failed each time (Dataset001/Dataset010
  succeeded at 700G; EMPIAR-10989/Dataset049 needed the 800G rerun, scoped to just those two by
  commenting out the already-done datasets in the `DATASETS` array).
- ❌ **DeePiCt + membrain-seg on DS001, EMPIAR-10989, and Dataset049** — only DS010 done. Need
  `baselines/submit_{deepict,membrain}_ds001.sbatch` / `..._ds989.sbatch` / `..._ds049.sbatch`
  (same pattern as the existing DS010 scripts, new `DATALIST`/`OUT` paths), then the corresponding
  `submit_{deepict,membrain}_inference_ds001.sbatch` / `..._ds989.sbatch` / `..._ds049.sbatch` for
  CryoDINO-schema metrics. Note: both kits binarize labels (`> 0`) — Dataset049's 4-class remapped
  labels would need to be binarized same as the others, i.e. these baselines can only ever report a
  single foreground-vs-background number for Dataset049, not per-class, unlike CryoDINO/UNet/UNETR.
- **10% / 50% data, all methods, all 4 datasets:**
  - ✅ **CryoDINO (best-checkpoint)** — done, see §2.2.
  - 🔄 **UNet/UNETR** — `slurm_scripts/train_baselines_patches_{10,50}pct.sh` created and running
    (path bug fixed: `train.py`/`inference.py` live under `baselines/unet_unetr/`, the original
    100% script and these copies initially pointed at `baselines/train.py` directly).
  - ❌ **CryoDINO random-init** — `slurm_scripts/train_3dino_ft_random_init_patches_{10,50}pct.sh`
    created (`--mem=800G` from the start, same percent-suffixed-datalist convention), not yet
    submitted.
  - ❌ **DeePiCt / membrain-seg** — no percent flag in either kit, but confirmed the pre-built
    percent-suffixed datalists (`Dataset{...}_patches512_{10,50}percent_100_datalist.json`) have
    the same `validation`/`test` as the 100% file with only `training` shrunk (DS010 10%:
    `training=12, validation=4, test=5` — matches the 100% file's val/test counts). So these can
    be passed directly as the `<datalist.json>` arg to `run_datalist.sh`, no kit changes needed —
    just new `submit_{deepict,membrain}_<dataset>_{10,50}pct.sbatch` pointing `DATALIST` at the
    percent-suffixed file.

### 2.5 Scripts to submit

**DeePiCt + membrain-seg on DS001 / EMPIAR-10989 / Dataset049 (100%, new):**
```bash
# copy baselines/submit_deepict_ds010.sbatch -> submit_deepict_ds001.sbatch / submit_deepict_ds989.sbatch / submit_deepict_ds049.sbatch
# copy baselines/submit_membrain_ds010.sbatch -> submit_membrain_ds001.sbatch / submit_membrain_ds989.sbatch / submit_membrain_ds049.sbatch
# (swap DATALIST to Dataset001_.../Dataset989_.../Dataset049_..._patches512_100_datalist.json and OUT dir)
sbatch baselines/submit_deepict_ds001.sbatch
sbatch baselines/submit_deepict_ds989.sbatch
sbatch baselines/submit_deepict_ds049.sbatch
sbatch baselines/submit_membrain_ds001.sbatch
sbatch baselines/submit_membrain_ds989.sbatch
sbatch baselines/submit_membrain_ds049.sbatch
# then, same pattern as the existing DS010 inference scripts:
sbatch baselines/submit_deepict_inference_ds001.sbatch
sbatch baselines/submit_deepict_inference_ds989.sbatch
sbatch baselines/submit_deepict_inference_ds049.sbatch
sbatch baselines/submit_membrain_inference_ds001.sbatch
sbatch baselines/submit_membrain_inference_ds989.sbatch
sbatch baselines/submit_membrain_inference_ds049.sbatch
```

**CryoDINO (both backbones) at 10%/50%** — edit `DATASET_PERCENT=100` → `10`/`50` in place
(per-dataset `j1`=DS001, `j2`=DS010, `j3`=EMPIAR-10989, `j4`=Dataset049) and resubmit:
```bash
sbatch slurm_scripts/train_3dino_ft_b200_highres128_6249_patches_j1.sh   # DS001, highres128
sbatch slurm_scripts/train_3dino_ft_b200_highres128_6249_patches_j2.sh   # DS010, highres128
sbatch slurm_scripts/train_3dino_ft_b200_highres128_6249_patches_j3.sh   # EMPIAR-10989, highres128
sbatch slurm_scripts/train_3dino_ft_b200_highres128_6249_patches_j4.sh   # Dataset049, highres128
sbatch slurm_scripts/train_3dino_ft_b200_highres_6249_patches_j1.sh      # DS001, highres112
sbatch slurm_scripts/train_3dino_ft_b200_highres_6249_patches_j2.sh      # DS010, highres112
sbatch slurm_scripts/train_3dino_ft_b200_highres_6249_patches_j3.sh      # EMPIAR-10989, highres112
sbatch slurm_scripts/train_3dino_ft_b200_highres_6249_patches_j4.sh      # Dataset049, highres112
```
(`DATASET_PERCENT=100` at line 45/44 in each; need to edit + resubmit once for 10% and once for
50%, restoring to 100 afterward if the same script is needed for other purposes — ❓ confirm
whether to run these sequentially re-editing in place, or keep separate copies.)

**CryoDINO random-init re-run (100%, needed) + 10%/50% (new):**
```bash
# DATASET_PERCENT=100 already set by default (line 38) — just resubmit as-is for the 100% re-run
sbatch slurm_scripts/train_3dino_ft_random_init_patches.sh
# then edit DATASET_PERCENT -> 10 / 50 in place and resubmit for each
```

**UNet/UNETR at 10%/50%** — edit `--dataset_percent 100` → `10`/`50` in place (line 92):
```bash
sbatch slurm_scripts/train_baselines_patches.sh
```

**DeePiCt/membrain-seg at 10%/50%, all 4 datasets** — new scripts following the same naming
pattern, `DATALIST` pointed at the percent-suffixed datalist:
```bash
# e.g. submit_deepict_ds010_10pct.sbatch, submit_membrain_ds010_10pct.sbatch, etc. — 4 datasets x
# 2 kits x 2 percentages = 16 new scripts, each a copy of the existing ds010 100% script with
# DATALIST swapped to the matching *_{10,50}percent_100_datalist.json
sbatch baselines/submit_deepict_ds001_10pct.sbatch
sbatch baselines/submit_deepict_ds001_50pct.sbatch
# ... (repeat for ds010, ds989, ds049, and membrain)
```

---

## 3. Detection benchmark

**Datasets:** CZI (Dataset440_CZII_10440, 6-class particle picking), BYU (bacterial flagellar
motor, single class).

**Methods:** CryoDINO (ViTAdapterUNETR detection head, B200 backbone), CryoDINO random-init,
DeepETPicker, ResNetFPN (`--segmentation-head ResNetFPN` in `detection3d.py` — a from-scratch
ResNet34-FPN trained under the **exact same loss** as CryoDINO: `object_detection_loss`/
`TaskAlignedAssigner`/`decode_detections_with_nms`, only the backbone differs).

> **Not** the MONAI RetinaNet baseline (`baselines/detection3d_monai/`) — that one trains under
> MONAI's own anchor-based RetinaNet loss (ATSS matcher + cls/box-regression loss), not ours, so
> it isn't a true same-loss comparison. Deliberately excluded from the main comparison for that
> reason; kept in the repo as a secondary/exploratory baseline, not reported here.

### 3.1 Datalists — ✅ CONFIRMED real (not placeholder), from `manuscript/dataset_split.csv`

Both honest train/val/test splits now exist, pre-built per data percentage (val/test fixed size,
only training shrinks), and confirmed to carry real ground-truth points (not the old all
`-100` placeholder we found earlier in the original shipped `czi_100_datalist.json`):

| Dataset | % | Datalist path | `--dataset-name` | `--base-data-dir` | Train patches | Train tomos | Val | Test |
|---|---|---|---|---|---|---|---|---|
| CZI | 100 | `.../Dataset440_CZII_10440_detection_patches_128/czi_100_datalist.json` | `czi` | `.../Dataset440_CZII_10440_detection_patches_128` | 1134 | 7 | 121 | 364 |
| CZI | 50 | `.../Dataset440_CZII_10440_detection_patches_128/czi_50percent/czi_100_datalist.json` | `czi` | `.../czi_50percent` | 648 | 4 | 121 | 364 |
| CZI | 10 | `.../Dataset440_CZII_10440_detection_patches_128/czi_10percent/czi_100_datalist.json` | `czi` | `.../czi_10percent` | 162 | 1 | 121 | 364 |
| BYU | 100 | `.../byu_detection_patches_128/byu_100_datalist.json` | `byu` | `.../byu_detection_patches_128` | 533,162 | 453 | 97 | 98 |
| BYU | 50 | `.../byu_detection_patches_128/byu_50percent/byu_100_datalist.json` | `byu` | `.../byu_50percent` | 275,478 | 227 | 97 | 98 |
| BYU | 10 | `.../byu_detection_patches_128/byu_10percent/byu_100_datalist.json` | `byu` | `.../byu_10percent` | 51,618 | 46 | 97 | 98 |

(all paths rooted at `/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets/downstream_detection/`;
note every percent variant keeps the literal filename `..._100_datalist.json` — the percent lives
in the directory name, not the file name, matching how `loaders.py` builds
`{base-data-dir}/{dataset-name}_100_datalist.json` unconditionally.)

### 3.2 Status matrix

| Method | CZI 10/50/100% | BYU 10/50/100% |
|---|---|---|
| CryoDINO (ViTAdapterUNETR, B200) | ❌/❌/❌ (post-fix, never run) | ❌/❌/❌ |
| CryoDINO random-init | ❌/❌/❌ | ❌/❌/❌ |
| DeepETPicker | ❌/❌/🟡 (10-epoch verification only) | ❌/❌/❌ |
| ResNetFPN (same-loss scratch) | ❌/❌/🟡 (matched-run verification only, see below) | ❌/❌/❌ |

Nothing in this section has a real benchmark number yet — but all the data infrastructure now
exists for CryoDINO/random-init/ResNetFPN on both datasets at all 3 percentages (ResNetFPN runs
through the same `detection3d.py` entry point, just a different `--segmentation-head`). DeepETPicker
is still CZI/Dataset440-only (hardcoded 6-class sigma map, `Dataset440_CZII_10440` paths) — BYU
support needs new kit configuration (single "motor" class, 6.5 Å/voxel) before it can run there.

The one existing ResNetFPN number (from the `detection-debug` import, matched run on CZI, val
`TS_69_2`/test `TS_73_6`, 3000 iters, bs 4, lr 1e-4 — **not** the full 30k-iteration budget) was:
CryoDINO frozen ViTAdapterUNETR test F4 **0.780** vs. ResNetFPN from-scratch test F4 **0.700** —
promising, but needs a real full-length re-run (same 100×300 budget as everything else) before
it's a reportable number.

### 3.3 Scripts to submit

**CryoDINO detection (CZI)** — `slurm_scripts/train_3dino_ft_h100_detection_czi.sh` already points
at the B200 checkpoint/`cryodino` env (fixed earlier this session). Edit `BASE_DATA_DIR`/
`DATASET_NAME` in place per the table above for each percent, keep `--dataset-percent 100` (the
directory itself is already the subsampled one, no further runtime subsampling needed):
```bash
sbatch slurm_scripts/train_3dino_ft_h100_detection_czi.sh
```

**CryoDINO detection (BYU)** — ❌ script doesn't exist yet. Needs a new
`slurm_scripts/train_3dino_ft_detection_byu.sh`, copied from the CZI one with
`DATASET_NAME="byu"` and `BASE_DATA_DIR` pointed at the BYU paths above (`loaders.py` already
branches on `dataset_name == 'byu'` → 1 class vs `'czi'` → 6 classes, so no code change needed,
just a new SLURM script).

**CryoDINO random-init (CZI + BYU)** — ❌ no detection random-init script exists yet. Segmentation's
convention (`train_3dino_ft_random_init_patches.sh`) sets `PRETRAINED_WEIGHTS=""` so
`load_pretrained_weights` fails to find a checkpoint and falls back to random init (the original,
pre-fix behavior — we deliberately did **not** take `detection-debug`'s "fail loudly on bad
weights" change, so this fallback path is still live for detection too). Needs 2 new scripts
(CZI + BYU), each a copy of the respective detection script above with `PRETRAINED_WEIGHTS=""`.

**DeepETPicker (CZI only, all 3 percentages)** — point `DATALIST`/`OUT` at each percent's path from
the table:
```bash
sbatch baselines/deepetpicker_kit/submit.sbatch   # edit DATALIST/OUT per percent, or parameterize
```
BYU: ❌ needs kit adaptation first (single-class sigma/diameter config, new `--tomo-dir`/`--csv`
equivalent for BYU's raw data format).

**ResNetFPN (same-loss scratch baseline, CZI + BYU, all 3 percentages)** — same entry point/script
as CryoDINO detection above, just swap the head:
```bash
# CZI — edit BASE_DATA_DIR/DATASET_NAME per percent as above, plus:
#   --segmentation-head ResNetFPN   (instead of ViTAdapterUNETR)
#   --pretrained-weights unused for this head (no DINO backbone built at all)
sbatch slurm_scripts/train_3dino_ft_h100_detection_czi.sh
# BYU — same new script as CryoDINO detection (BYU) above, also with --segmentation-head ResNetFPN
```
No separate kit/file needed — it's a flag on the same training script, so BYU support comes for
free once the BYU detection SLURM script exists (unlike DeepETPicker, which needs real kit work).

### 3.4 Still needed / open questions

- ❌ BYU adaptation for DeepETPicker — real engineering work (single-class sigma/diameter config,
  new data-format converter), not just a resubmit with different paths. ResNetFPN doesn't have
  this problem (see above).
- ❓ Confirm the CZI/BYU datalists' points use the same coordinate/sigma convention
  `detection3d.py` expects (same as segmentation's dataset-specific quirks) — not verified in
  this session, only structural shape (train/val/test counts) was confirmed from the CSV.
