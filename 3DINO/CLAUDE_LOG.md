## Claude Response (Sat 13 Jun 2026 01:05:43 PM EDT)
Fixed and verified. Here's what was wrong and what I changed.

## The bug

`CZIDetectionMetrics` indexes everything by integer class ID, but its `PARTICLE_NAMES`/`PARTICLE_RADII_ANG`/`WEIGHTS` were in **official Kaggle order** (`0=ferritin, 1=Beta-amylase…`), while the model is trained on **alphabetical** class IDs (`load_detection_annotations` does `sorted()` → `0=Beta-amylase … 4=ferritin`). So inside `_f4_at_threshold`, class `cls_idx` selected the right predictions but applied the **wrong particle's radius and weight**:

| class id | particle (model) | was using | should use |
|---|---|---|---|
| 0 | Beta-amylase | r=60, w=1 | r=65, **w=0** |
| 2 | Thyroglobulin | r=90, w=2 | r=130, w=2 |
| 4 | ferritin complex | r=130, w=2 | **r=60, w=1** |

This corrupted per-class F4, the weighting (the excluded `w=0` class landed on the wrong particle), and the NMS radii in `detection3d.py` / the inference script — and it affected **checkpoint selection during training**, since `best_model.pth` is chosen by this metric. (I was wrong earlier that the aggregate "still matches" — wrong radii change TP/FP/FN, so the aggregate was off too.)

## The fix

**`metrics.py`** — replaced the three hand-ordered lists with a single name→`(radius, weight)` dict and derived the ordering via `sorted()`, the *same call* the data pipeline uses. This makes drift structurally impossible: the metric's class IDs are now guaranteed to match the model's output channels.

**`scripts_debug/compare_czii_metrics.py`** — its `OFFICIAL_NAMES` mapped class index → name in official order (which is why it never caught the bug). Reordered to alphabetical so the equivalence test stays meaningful.

## Verification

- `compare_czii_metrics.py`: all 3 checks PASS — constants match, and aggregate F-beta is numerically identical to the official scorer (max diff 2.8e-17 over 800 comparisons).
- Per-name radius/weight cross-checked against `official_metrics_czii.py`: correct.
- Order confirmed alphabetical; total weight unchanged (7).
- Both downstream consumers (`detection3d.py`, `detection3d_inference.py`) index `PARTICLE_RADII_ANG` by class ID, so they're auto-corrected — no edits needed; all three files compile.

## What this means for you

- Going forward, training's reported val F4 (and which checkpoint is saved as best) will **change** — they're now correct. Old `results.json` numbers from prior runs are not comparable to new ones.
- I'd recommend re-running training, or at minimum re-running step-6 inference, since `best_model.pth` was selected under the buggy metric.

One thing I did **not** touch: the prediction CSV naming was already correct (alphabetical), so those outputs are unaffected.
