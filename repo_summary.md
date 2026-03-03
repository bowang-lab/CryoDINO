# Repository Overview: CryoET Foundation Model

This is a **3DINO** (3D self-supervised learning) framework adapted for **Cryo-Electron Tomography (CryoET)** data. The goal is to build a foundation model that learns general representations from unlabeled 3D tomograms through self-supervised learning.

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `3DINO/dinov2/` | Core framework - training, models, data loading, losses, evaluation |
| `preprocessing/` | Data preparation - subtomogram extraction, JSON metadata generation, deconvolution |
| `slurm_scripts/` | SLURM job submission scripts for cluster training |

## Key Components

### Training Pipeline
- `train/train3d.py` - Main distributed training entry point
- `train/ssl_meta_arch.py` - Student/teacher architecture with DINO, iBOT, KoLeo losses

### Model
- `models/vision_transformer.py` - 3D Vision Transformer (ViT-Large: 24 blocks, 1024 dim)
- `layers/patch_embed3d.py` - 3D patch embedding (16³ patches)

### Data
- `data/loaders.py` - Loads from JSON metadata files
- `data/augmentations.py` - 3D random crops, rotations, intensity transforms

### Evaluation
- `eval/segmentation3d.py` - Fine-tune for segmentation (UNETR, ViTAdapter heads)
- `eval/linear3d.py` - Fine-tune for classification

## Technologies

- **PyTorch 2.0** with DDP/FSDP for distributed training
- **MONAI** for medical imaging transforms and data loading
- **xFormers** for memory-efficient attention
- Requires **A100-80GB GPUs** (4 recommended)

## Data Flow

```
Raw tomograms → subtomograms_generation.py → 128³ patches (.pt)
       ↓
create_pretrain_json.py → pretrain.json (metadata)
       ↓
Training: 3D augmentations → ViT encoder → DINO/iBOT losses
       ↓
Fine-tuning: Pretrained backbone → segmentation/classification head
```

## Training Configuration

- **Batch size:** 128 per GPU (512 total on 4 GPUs)
- **Epochs:** 100 standard
- **Learning rate:** 0.002 base with cosine schedule
- **Optimizer:** AdamW
- **Augmentations:** Global crops (96³), local crops (48³), flips, rotations, intensity transforms

## Loss Functions

- **DINO Loss:** Contrastive learning on class tokens with Sinkhorn-Knopp centering
- **iBOT Loss:** Masked patch prediction (20-75% mask ratio)
- **KoLeo Loss:** Prototype diversity regularizer
