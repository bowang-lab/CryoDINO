# 3DINO Loss Functions Summary

This document explains each loss function used in the 3DINO self-supervised learning framework.

---

## 1. DINO Loss (Local & Global Crops)
**File:** `3DINO/dinov2/loss/dino_clstoken_loss.py`

### What it does:
Compares the **[CLS] token** (global representation) between student and teacher networks using **cross-entropy**.

### How it's computed:
```python
# Teacher output is centered and softmax-sharpened
teacher_softmax = softmax((teacher_output - center) / teacher_temp)

# Student output is softmax with higher temperature
student_log_softmax = log_softmax(student_output / student_temp)

# Cross-entropy loss
loss = -sum(teacher_softmax * student_log_softmax)
```

### What it minimizes:
- Makes **student features match teacher features** for different augmented views of the same image
- The teacher is an exponential moving average (EMA) of the student → self-distillation
- **Centering** prevents collapse (all outputs being the same)
- **Sinkhorn-Knopp** normalization ensures uniform cluster assignment

### Local vs Global:
| Loss | Student Input | Teacher Input |
|------|--------------|---------------|
| `dino_local_crops_loss` | 8 small (48³) crops | 2 large (96³) crops |
| `dino_global_crops_loss` | 2 large (96³) crops | 2 large (96³) crops (cross-view) |

**Goal:** Local crops should have similar representations to global crops of the same image.

---

## 2. iBOT Loss (Masked Patch Prediction)
**File:** `3DINO/dinov2/loss/ibot_patch_loss.py`

### What it does:
**Masked Image Modeling** - predicts features of **masked patches** from visible patches.

### How it's computed:
```python
# Mask random patches in student input (20-75% of patches)
# Student sees masked image, teacher sees full image

# Cross-entropy between student's prediction and teacher's features for masked patches
loss = -sum(teacher_patch_tokens * log_softmax(student_patch_tokens / temp))
loss = loss * mask  # only for masked patches
```

### What it minimizes:
- Student must **reconstruct masked patch features** using context from visible patches
- Similar to BERT/MAE but in feature space, not pixel space
- Forces the model to learn **local spatial relationships**

**Goal:** Learn fine-grained patch-level representations, not just global [CLS] token.

---

## 3. KoLeo Loss (Uniformity Regularizer)
**File:** `3DINO/dinov2/loss/koleo_loss.py`

### What it does:
**Kozachenko-Leonenko entropy estimator** - encourages features to be **uniformly spread** in the embedding space.

### How it's computed:
```python
# Normalize features to unit sphere
features = normalize(student_output)

# Find nearest neighbor for each sample
nn_idx = argmax(features @ features.T)  # excluding self

# Compute distance to nearest neighbor
distances = ||features - features[nn_idx]||

# Maximize distances (minimize negative log)
loss = -log(distances).mean()
```

### What it minimizes:
- **Prevents collapse** where all representations cluster together
- Encourages **uniform distribution** on the hypersphere
- Maximizes the **distance to nearest neighbor**

**Goal:** Ensure diverse representations - each sample should be different from others.

---

## Summary Table

| Loss | Level | What it Learns | Prevents |
|------|-------|----------------|----------|
| **DINO Local** | Global ([CLS]) | View invariance (small→large) | - |
| **DINO Global** | Global ([CLS]) | View invariance (large↔large) | Collapse (via centering) |
| **iBOT** | Patch-level | Local spatial context | - |
| **KoLeo** | Global ([CLS]) | Feature diversity | Collapse (via uniformity) |

---

## How They Work Together

```
Input Image
    ↓
┌─────────────────────────────────────────┐
│  Student (learns)    Teacher (EMA)      │
│       ↓                   ↓             │
│   [CLS] token ←──DINO──→ [CLS] token    │  ← View consistency
│       ↓                   ↓             │
│  Patch tokens ←──iBOT──→ Patch tokens   │  ← Masked prediction
│       ↓                                 │
│   [CLS] token ←──KoLeo──→ (uniformity)  │  ← Prevent collapse
└─────────────────────────────────────────┘
```

---

## Total Loss Computation

The **total_loss** is a weighted sum of all components:

```python
total_loss = (dino_loss_weight * dino_local_crops_loss)
           + (dino_loss_weight * dino_global_crops_loss)
           + (koleo_loss_weight * koleo_loss)
           + (ibot_loss_weight * ibot_loss)
```

Default weights (from config):
- `dino_loss_weight`: 1.0
- `ibot_loss_weight`: 1.0
- `koleo_loss_weight`: 0.1

---

## References

- DINO: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- iBOT: [Image BERT Pre-training with Online Tokenizer](https://arxiv.org/abs/2111.07832)
- KoLeo: [Spreading vectors for similarity search](https://arxiv.org/abs/1806.03198)
- DINOv2: [Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
