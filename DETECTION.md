# CZII Detection

Run from `/cluster/home/t129616uhn/projects/CryoDINO` on `uhn-login`.

## Paths

```
DATA=/cluster/projects/pmccaiteam/reza/projects/cryoet/datasets/downstream_detection
OUT=/cluster/projects/pmccaiteam/reza/projects/cryoet/experiments/finetuning_detection
```

### Datalists (image + point labels, keys: training / validation / test)

| Split | Datalist |
|---|---|
| 100% | `$DATA/Dataset440_CZII_10440_detection_patches_128/czi_100_datalist.json` |
| 50%  | `$DATA/Dataset440_CZII_10440_detection_patches_128/czi_50percent/czi_100_datalist.json` |
| 10%  | `$DATA/Dataset440_CZII_10440_detection_patches_128/czi_10percent/czi_100_datalist.json` |

### Train (128³ patches, `.pt`)

| Split | Tomograms | Patches |
|---|---|---|
| 100% | TS_5_4, TS_69_2, TS_6_4, TS_6_6, TS_73_6, TS_86_3, TS_99_9 | 1134 |
| 50%  | TS_69_2, TS_6_4, TS_6_6, TS_73_6 | 648 |
| 10%  | TS_69_2 | 162 |

```
images: $DATA/Dataset440_CZII_10440_detection_patches_128/images/
labels: "points" field in datalist
```

### Val / Test (whole tomograms, same for 100/50/10%)

| Split | Images | Labels | N |
|---|---|---|---|
| Val  | `$DATA/czi_dataset/Dataset445_CZII_10445/imagesTr/` | `$DATA/czi_dataset/Dataset445_CZII_10445/labelsTr/`, `point_annotations.csv` | 121 |
| Test | `$DATA/czi_dataset/Dataset446_CZII_10446/imagesTr/` | `$DATA/czi_dataset/Dataset446_CZII_10446/labelsTr/`, `point_annotations.csv` | 364 |

## Commands

`P` = `100`, `50` or `10`.

```bash
# Frozen CryoDINO (ViT-Adapter + UNETR)
sbatch -J det-cryodino-czi-$P -t 3-00:00:00 --mem=76G --cpus-per-task=12 --export=ALL,METHOD=cryodino,DATASET=czi,PERCENT=$P,NUM_WORKERS=8 slurm_scripts_detection/train_detection.sh

# Random init (ViT-Adapter + UNETR, trained from scratch)
sbatch -J det-random_init-czi-$P -t 3-00:00:00 --mem=76G --cpus-per-task=12 --export=ALL,METHOD=random_init,DATASET=czi,PERCENT=$P,NUM_WORKERS=8 slurm_scripts_detection/train_detection.sh

# ResNet34-FPN from scratch
sbatch -J det-resnetfpn-czi-$P -t 3-00:00:00 --mem=76G --cpus-per-task=12 --export=ALL,METHOD=resnetfpn,DATASET=czi,PERCENT=$P,NUM_WORKERS=8 slurm_scripts_detection/train_detection.sh

# Frozen + LoRA r16 (qkv, proj), lr 1e-4
sbatch -J det-cryodino-czi-$P-lora16 -t 3-00:00:00 --mem=90G --cpus-per-task=12 --exclude=node164 --export=ALL,METHOD=cryodino,DATASET=czi,PERCENT=$P,NUM_WORKERS=8,LEARNING_RATE=1e-4,LORA_RANK=16,LORA_LR=1e-4 slurm_scripts_detection/train_detection_hyp.sh

# Frozen + adapter_fpn, lr 1e-3, grad clip 3.0
sbatch -J det-cryodino-czi-$P-afpn-clip -t 3-00:00:00 --mem=90G --cpus-per-task=12 --exclude=node164 --export=ALL,METHOD=cryodino,DATASET=czi,PERCENT=$P,NUM_WORKERS=8,DECODER_FUSION=adapter_fpn,LEARNING_RATE=1e-3,CLIP_GRAD=3.0 slurm_scripts_detection/train_detection_hyp.sh
```

100% only:

```bash
# Last ViT block unfrozen
sbatch -J det-cryodino_last1-czi-100 -t 3-00:00:00 --mem=76G --cpus-per-task=12 --export=ALL,METHOD=cryodino_last1,DATASET=czi,PERCENT=100,NUM_WORKERS=8 slurm_scripts_detection/train_detection.sh

# Full fine-tune, lr 1e-4
sbatch -J det-cryodino_ft-czi-100 -t 3-00:00:00 --mem=280G --cpus-per-task=32 --export=ALL,METHOD=cryodino_ft,DATASET=czi,PERCENT=100,NUM_WORKERS=24 slurm_scripts_detection/train_detection.sh

# Full fine-tune, lr 1e-5
sbatch -J det-cryodino_ft-czi-100-lr1e-5 -t 3-00:00:00 --mem=280G --cpus-per-task=32 --export=ALL,METHOD=cryodino_ft,DATASET=czi,PERCENT=100,NUM_WORKERS=24,LEARNING_RATE=1e-5 slurm_scripts_detection/train_detection_hyp.sh

# Frozen, lr 1e-5
sbatch -J det-cryodino-czi-100-lr1e-5 -t 3-00:00:00 --mem=76G --cpus-per-task=12 --export=ALL,METHOD=cryodino,DATASET=czi,PERCENT=100,NUM_WORKERS=8,LEARNING_RATE=1e-5 slurm_scripts_detection/train_detection_hyp.sh

# Frozen + adapter_fpn, lr 1e-4
sbatch -J det-cryodino-czi-100-adapter_fpn -t 3-00:00:00 --mem=190G --cpus-per-task=28 --export=ALL,METHOD=cryodino,DATASET=czi,PERCENT=100,NUM_WORKERS=20,LEARNING_RATE=1e-4,DECODER_FUSION=adapter_fpn slurm_scripts_detection/train_detection_hyp.sh
```

## Outputs (test results)

Each run directory holds `results.json` (`test_f4`, `test_per_cls_f4`, `best_val_f4`, `best_val_thresholds`), `best_model.pth`, `config.yaml` and `logs/`. Per-tomogram test predictions are not saved.

| Model | 100% | 50% | 10% |
|---|---|---|---|
| Frozen CryoDINO | `$OUT/cryodino_b200_highres128_6249_czi_100pct_vit_adapter` | `$OUT/cryodino_b200_highres128_6249_czi_50pct_vit_adapter` | `$OUT/cryodino_b200_highres128_6249_czi_10pct_vit_adapter` |
| Random init | `$OUT/random_init_czi_100pct_vit_adapter` | `$OUT/random_init_czi_50pct_vit_adapter` | `$OUT/random_init_czi_10pct_vit_adapter` |
| ResNet34-FPN | `$OUT/resnetfpn_scratch_czi_100pct` | `$OUT/resnetfpn_scratch_czi_50pct` | `$OUT/resnetfpn_scratch_czi_10pct` |
| Frozen + LoRA r16 | `$OUT/cryodino_b200_highres128_6249_czi_100pct_vit_adapter_lora16_loralr1e-4_lr1e-4_bs4_ep100x300_wu3000` | `$OUT/cryodino_b200_highres128_6249_czi_50pct_vit_adapter_lora16_loralr1e-4_lr1e-4_bs4_ep100x300_wu3000` | `$OUT/cryodino_b200_highres128_6249_czi_10pct_vit_adapter_lora16_loralr1e-4_lr1e-4_bs4_ep100x300_wu3000` |
| Frozen + adapter_fpn, lr 1e-3, clip 3.0 | `$OUT/cryodino_b200_highres128_6249_czi_100pct_vit_adapter_adapter_fpn_clip3.0_lr1e-3_bs4_ep100x300_wu3000` | `$OUT/cryodino_b200_highres128_6249_czi_50pct_vit_adapter_adapter_fpn_clip3.0_lr1e-3_bs4_ep100x300_wu3000` | `$OUT/cryodino_b200_highres128_6249_czi_10pct_vit_adapter_adapter_fpn_clip3.0_lr1e-3_bs4_ep100x300_wu3000` |
| Last ViT block unfrozen | `$OUT/cryodino_last1_b200_highres128_6249_czi_100pct_vit_adapter` | | |
| Full fine-tune, lr 1e-4 | `$OUT/cryodino_ft_b200_highres128_6249_czi_100pct_vit_adapter` | | |
| Full fine-tune, lr 1e-5 | `$OUT/cryodino_ft_b200_highres128_6249_czi_100pct_vit_adapter_lr1e-5_bs4_ep100x300_wu3000` | | |
| Frozen, lr 1e-5 | `$OUT/cryodino_b200_highres128_6249_czi_100pct_vit_adapter_lr1e-5_bs4_ep100x300_wu3000` | | |
| Frozen + adapter_fpn, lr 1e-4 | `$OUT/cryodino_b200_highres128_6249_czi_100pct_vit_adapter_adapter_fpn_lr1e-4_bs4_ep100x300_wu3000` | | |

## DeepETPicker (baseline)

Each directory holds `metrics_czi.json`, `test/` (per-tomogram `*.coords` predictions), `validation/`, `_checkpoint.txt` and `_timing.json`.

| Split | Results |
|---|---|
| 100% | `/cluster/projects/pmccaiteam/charles/deepetpicker/results/czii_100/` |
| 50%  | `/cluster/projects/pmccaiteam/charles/deepetpicker/results/czii_50/` |
| 10%  | `/cluster/projects/pmccaiteam/charles/deepetpicker/results/czii_10/` |
