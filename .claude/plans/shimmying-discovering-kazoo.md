# Fix: `points` is a Python list, not ndarray, in detection augmentations

## Context
Detection fine-tuning (`train_3dino_ft_h100_detection_czi.sh`) crashes in the dataloader:

```
File ".../detection_3d/augmentations.py", line 89, in __call__
    valid = pts[:, 3] >= 0
TypeError: list indices must be integers or slices, not tuple
```

**Root cause:** The CZI datalist (`czi_100_datalist.json`) stores each sample's `"points"`
as a JSON array-of-arrays. MONAI's `PersistentDataset` passes that through unchanged, so
`d["points"]` is a Python `list`. `RandFlipWithPointsd` / `RandRotate90WithPointsd` do
`pts = d["points"].copy()` (a plain list copy) and then index it as `pts[:, 3]`, which is
only valid for an `np.ndarray`. The augmentation docstring already assumes
`"points": np.ndarray [N, 5]` — the data just never gets converted to satisfy that contract.

Verified: `json.load(...)['training'][0]['points']` → `list` of 5-element `list`s.

## Change
Add one deterministic load-stage transform in `make_transforms` (file:
`3DINO/dinov2/eval/detection_3d/augmentations.py`, the `_load` list ~line 213) to coerce
`points` to an `Nx5 float32` ndarray:

```python
Lambdad(keys=["points"], func=lambda p: np.asarray(p, dtype=np.float32).reshape(-1, 5)),
```

Placed right after the image `Lambdad`. Rationale:
- It is deterministic, so `PersistentDataset` caches it (no per-iteration cost).
- Applies to train **and** val/test transforms (both built from `_load`), so the collate
  function's `d["points"].shape[0]` (detection3d.py:31) and `torch.as_tensor(d["points"])`
  (detection3d.py:34) are also fed a real array.
- `.reshape(-1, 5)` keeps shape `(0, 5)` for patches with zero points (otherwise
  `np.asarray([])` → shape `(0,)` and `pts[:, 3]` would fail again).

Status: this edit was already applied to `augmentations.py` before plan mode engaged.

## Verification
- Re-run `slurm_scripts/train_3dino_ft_h100_detection_czi.sh` on the cluster; training loop
  should pass the first dataloader batch without the `TypeError`.
- Quick local smoke test (no GPU needed) on the data mount:
  ```python
  from dinov2.eval.detection_3d.augmentations import make_transforms
  import json
  d = json.load(open('.../czi_100_datalist.json'))['training'][0]
  tr, _ = make_transforms()
  out = tr(dict(d))            # should not raise; out['points'] is an ndarray/tensor
  ```
- Since the `_load` stage changed, the existing cache dir (`CACHE_DIR`) must be cleared so
  the new deterministic transform is re-cached — the slurm script already does
  `rm -rf "$CACHE_DIR"` before each run, so no extra action needed.
