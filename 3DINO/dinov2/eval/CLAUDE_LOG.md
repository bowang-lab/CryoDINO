## Claude Response (Sat 13 Jun 2026 08:58:33 AM EDT)
Done. Here's the final state of the verification and the changes made.

## What I verified

`sliding_window_accumulate` and its two call sites (`detection_val_iter` → in-training validation at overlap 0.5, and final test/inference at overlap 0.75). The **accumulation algorithm itself is correct**: tiling covers the volume, raw logits are averaged then sigmoid applied once at decode, offset averaging is valid (anchors are a function of the global cell only; offsets are bounded `tanh()*2`), per-tile normalization matches training, and the counter/zero-mask is right.

I found two real bugs upstream of the math that determine whether it produces correct results at all.

## Changes made

**Bug 1 — fatal loader mismatch (fixed + verified).** Val/test datalist entries are raw `.nii.gz` tomograms, but `make_transforms._load` used `torch.load()` (a `.pt` reader) → `UnpicklingError` on every val/test item. Added `_load_detection_image` in `augmentations.py` that branches on extension: nibabel + z-score for NIfTI, `torch.load` for `.pt`. Verified on the real files (NIfTI → z-scored `[1,630,630,184]`; `.pt` still `[1,128,128,128]`).

**Bug 2 — axis swap (confirmed, then patched per your call).** `anchors_for_offsets_feature_map` stacked anchors so channel 0 tracked the **last** axis (Z, max 183) while GT is `[x,y,z]` with channel 0 = first axis (X, up to ~629). Invisible on cubic 128³ training patches, broken on non-cubic val/test volumes. A standalone test (`scripts_debug/test_decode_axis_swap.py`) reproduced it (`[81,301,601]` vs expected `[601,301,81]`), and confirms the fix (now `[601,301,81]`, NO swap). Changed the stack to `[a0,a1,a2]` so channel 0 == X, and corrected the stale `off_head` channel comment.

All three edited files pass `py_compile`.

## Important caveat (your "no retrain" choice)

`decode_detections` is shared by the training loss and inference, so the current checkpoints were trained under the old swapped convention. **The patched code is correct, but inference on existing checkpoints will not give valid detections until you retrain** — the new decode and the old weights disagree. The fix is ready for the next training run whenever you schedule it.

Also two non-blocking notes: a latent fragility in `sliding_window_accumulate` (boundary tile start `dim - ps` is only stride-aligned for even dims — fine for 630/630/184, worth an `assert` for odd future volumes), and several stale comments in `loss.py`/`detection3d.py` claiming sigmoid is applied per-tile (it isn't — logits are averaged, sigmoid once at decode). Want me to clean up those comments and add the assert?
