"""
mix_patches_augmentation_cryodino.py
Author: Ahmadreza Attarpour

Augments CryoDINO fine-tuning patches (.pt format) to balance foreground class
representation across the training set while also augmenting every class.

Motivation:
-----------
Cryo-ET segmentation datasets are heavily class-imbalanced: rare organelles/structures
may appear in only a handful of patches, causing the model to ignore them (e.g. Class 1
in DS049 consistently achieves ~0.00 Dice). Standard within-patch augmentation cannot
fix this — we need to synthetically *create* new training samples that are rich in the
underrepresented classes.

Balancing strategy:
-------------------
We balance based on total foreground VOXELS per class. Every class — including the
dominant one — is augmented so that:

  1. V_max  = max(V[c]) across all foreground classes
  2. target = V_max × target_multiplier  (default 2.0 → all classes reach 2× the original peak)
  3. deficit[c] = target - V[c]          (all classes have a positive deficit)
  4. n_patches[c] = ceil(deficit[c] / avg_voxels_per_source_patch[c])

This means:
  - Dominant class:  gets augmented to ~2× its original count  (diversity + more data)
  - Minority classes: get augmented by much more to close the gap to the same target
  - Final dataset:   all classes approach `target` voxels → balanced + enlarged

Algorithm per augmented patch:
------------------------------
  a. Randomly pick a source patch rich in the target class.
  b. Randomly sample a coord from the dilated foreground of the target class
     and crop a patch_size³ sub-volume starting from that coord.
  c. Randomly pick a near-empty patch as the background canvas.
  d. Find canvas voxels where label==0 (no fg) and the crop fits; randomly
     select one as the insertion location.
  e. SNR-match: scale = sqrt(noise_canvas_region / noise_source_bg) × rand_snr_weight,
     where rand_snr_weight ~ Uniform(0.5, 2.0). Preserves realistic brightness vs. background.
  f. Save augmented image and label as .pt files.

SNR definition (z-score space):
  SNR = E[image[label > 0]²] / E[image[label == 0]²]

Usage:
------
  python preprocessing/mix_patches_augmentation_cryodino.py \
      --datalist         /path/to/Dataset049_patches512_100_datalist.json \
      --output-dir       /path/to/augmented_049 \
      --num-classes      4    \
      --min-crop-size    128  \
      --target-multiplier 2.0 \
      --empty-threshold  0.02 \
      --source-threshold 0.001 \
      --dilation-iters   3    \
      --seed 42

Output:
-------
  <output-dir>/images/aug_c{class}_{idx:05d}.pt
  <output-dir>/labels/aug_c{class}_{idx:05d}.pt
  <output-dir>/<original_stem>_augmented.json   ← original training + augmented entries
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import binary_dilation
import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_patch(path):
    """Load a .pt patch. Returns float32 numpy array of shape (D, H, W)."""
    t = torch.load(path, map_location='cpu', weights_only=True).float()
    return t.numpy()


def save_patch(array, path):
    """Save a numpy array as a .pt file (no channel dim)."""
    torch.save(torch.from_numpy(array), path)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_patches(entries, num_classes):
    """
    Analyse all training patches.

    Returns:
      stats        : list of per-patch dicts
      total_voxels : np.ndarray (num_classes,) — total voxels per class
    """
    stats = []
    print(f"\nAnalysing {len(entries)} training patches ...")

    for i, entry in enumerate(tqdm.tqdm(entries)):
        img   = load_patch(entry['image'])
        label = load_patch(entry['label']).astype(np.int32)

        # Sanity check shapes
        if img.shape != label.shape:
            print(f"  [WARNING] patch {i}: image shape {img.shape} != label shape {label.shape} — skipping")
            continue

        total        = label.size
        class_counts = np.array([(label == c).sum() for c in range(num_classes)], dtype=np.int64)
        fg_count     = int(total - class_counts[0])

        # Unique classes present
        unique_cls = np.unique(label).tolist()

        stats.append({
            'image':           entry['image'],
            'label':           entry['label'],
            'shape':           img.shape,
            'img_min':         float(img.min()),
            'img_max':         float(img.max()),
            'fg_fraction':     fg_count / total,
            'class_counts':    class_counts,
            'class_fractions': class_counts / total,
            'unique_classes':  unique_cls,
        })

    all_counts   = np.stack([s['class_counts'] for s in stats])
    total_voxels = all_counts.sum(axis=0)

    print("\n── Dataset analysis ──")
    print(f"  Total patches   : {len(stats)}")
    print(f"  Patch shape     : {stats[0]['shape']}  (first patch)")
    print(f"  Image range     : [{np.min([s['img_min'] for s in stats]):.3f}, "
          f"{np.max([s['img_max'] for s in stats]):.3f}]  (across all patches)")
    print(f"  Unique label values across dataset: "
          f"{sorted(set(v for s in stats for v in s['unique_classes']))}")
    print()
    for c in range(num_classes):
        n_with    = (all_counts[:, c] > 0).sum()
        label_str = "background" if c == 0 else f"class {c}   "
        avg_per_patch = total_voxels[c] / len(stats)
        print(f"  {label_str}: {total_voxels[c]:>12,} voxels total  |  "
              f"present in {n_with}/{len(stats)} patches  |  "
              f"avg {avg_per_patch:.0f} vox/patch")

    return stats, total_voxels


def split_patches(stats, empty_threshold, source_threshold, num_classes):
    """
    Returns:
      empty_patches     : low-fg patches usable as background canvases
      source_patches[c] : patches with enough of class c to crop from
    """
    empty_patches  = [s for s in stats if s['fg_fraction'] < empty_threshold]
    source_patches = {
        c: [s for s in stats if s['class_fractions'][c] > source_threshold]
        for c in range(1, num_classes)
    }

    print("\n── Patch availability ──")
    print(f"  Empty canvases  : {len(empty_patches)}  (fg < {empty_threshold*100:.1f}%)")
    if empty_patches:
        print(f"    fg fractions  : min={min(s['fg_fraction'] for s in empty_patches):.4f}  "
              f"max={max(s['fg_fraction'] for s in empty_patches):.4f}")
    for c in range(1, num_classes):
        srcs = source_patches[c]
        print(f"  Source class {c}  : {len(srcs)}  (class frac > {source_threshold*100:.3f}%)")
        if srcs:
            avg_vox = np.mean([s['class_counts'][c] for s in srcs])
            print(f"    avg vox/patch : {avg_vox:.0f}")

    return empty_patches, source_patches


def print_augmentation_plan(total_voxels, num_classes, target_multiplier):
    """Print the universal fg target and per-class deficit before generation.

    target = sum(all fg voxels) * factor / num_fg_classes
    """
    total_fg = int(total_voxels[1:].sum())
    target   = int(np.ceil(total_fg * target_multiplier / (num_classes - 1)))

    print(f"\n── Augmentation plan ──")
    print(f"  Total fg voxels  : {total_fg:,}")
    print(f"  Target (sum×{target_multiplier:.1f}/N): {target:,}")
    print()
    for c in range(1, num_classes):
        deficit = max(0, target - int(total_voxels[c]))
        print(f"  Class {c}: current={total_voxels[c]:,}  deficit={deficit:,}")
    return target


# ─────────────────────────────────────────────────────────────────────────────
# Crop extraction
# ─────────────────────────────────────────────────────────────────────────────

def crop_around_class(image, label, class_id, min_crop_size, dilation_iters, rng):
    """
    Crop a sub-volume starting from a randomly selected coord within the dilated
    foreground of class_id.  Crop dimensions are random per axis (matching
    mix_patches_augmentation.py):
      dim_i ~ randint(min_crop_size, full_dim_i)

    Returns (crop_image, crop_label, success, info_dict).
    """
    fg_mask = (label == class_id)
    if fg_mask.sum() == 0:
        return None, None, False, "no fg voxels for class"

    D, H, W = image.shape
    if any(s < min_crop_size for s in (D, H, W)):
        return None, None, False, f"patch {image.shape} smaller than min_crop_size {min_crop_size}"

    # Random crop size per axis: randint(min_crop_size, full_dim)  ← matches original
    ch = rng.randint(min_crop_size, D + 1)
    cw = rng.randint(min_crop_size, H + 1)
    cd = rng.randint(min_crop_size, W + 1)

    dilated        = binary_dilation(fg_mask, structure=np.ones((3, 3, 3)), iterations=dilation_iters)
    dilated_coords = np.argwhere(dilated)

    # Randomly select a start coord from the dilated fg coords
    start_coord = dilated_coords[rng.randint(0, len(dilated_coords))]
    d0 = int(max(0, min(int(start_coord[0]), D - ch)))
    h0 = int(max(0, min(int(start_coord[1]), H - cw)))
    w0 = int(max(0, min(int(start_coord[2]), W - cd)))

    crop_img = image[d0:d0+ch, h0:h0+cw, w0:w0+cd].copy()
    crop_lbl = label[d0:d0+ch, h0:h0+cw, w0:w0+cd].copy()

    info = {
        'start_coord':     start_coord.tolist(),
        'starts':          [d0, h0, w0],
        'crop_shape':      crop_img.shape,
        'fg_in_crop':      int((crop_lbl == class_id).sum()),
        'classes_in_crop': np.unique(crop_lbl).tolist(),
    }
    return crop_img, crop_lbl, True, info


# ─────────────────────────────────────────────────────────────────────────────
# SNR-matched insertion
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr(image, label):
    """SNR = E[fg²] / E[bg²] in z-score space."""
    fg = image[label > 0].astype(np.float64)
    bg = image[label == 0].astype(np.float64)
    if fg.size == 0 or bg.size == 0:
        return 1.0
    return float(np.mean(fg ** 2) / (np.mean(bg ** 2) + 1e-12))


def insert_with_snr_match(canvas_img, canvas_lbl, crop_img, crop_lbl, rng,
                          rand_weight_range=(0.5, 2.0)):
    """
    Insert crop into canvas at a random location where canvas_lbl == 0 (no fg),
    with SNR matching identical to mix_patches_augmentation.py:
      scale = sqrt(noise_canvas_region / noise_source_bg) * rand_snr_weight
    Returns (augmented_image, augmented_label, info_dict).
    """
    D, H, W = canvas_img.shape
    ch, cw, cd = crop_img.shape  # may be non-cubic after random crop sizing

    fg_mask = crop_lbl > 0
    if fg_mask.sum() == 0:
        return canvas_img.copy(), canvas_lbl.copy(), {'skipped': 'no fg in crop'}

    # Find valid insertion locations: canvas label == 0 AND crop fits within bounds.
    # Subsample bg coords for efficiency (large patches have ~100M+ bg voxels).
    bg_coords = np.argwhere(canvas_lbl == 0)
    if len(bg_coords) > 50000:
        idx = rng.choice(len(bg_coords), size=50000, replace=False)
        bg_coords = bg_coords[idx]

    valid = [(int(d), int(h), int(w)) for d, h, w in bg_coords
             if int(d) + ch <= D and int(h) + cw <= H and int(w) + cd <= W]
    if not valid:
        return canvas_img.copy(), canvas_lbl.copy(), {'skipped': 'no valid bg insertion location'}
    d0, h0, w0 = valid[rng.randint(0, len(valid))]

    print(f"  insert@({d0}, {h0}, {w0})  crop_shape={crop_img.shape}  canvas={canvas_img.shape}")

    aug_img = canvas_img.copy()
    aug_lbl = canvas_lbl.copy()

    sl = (slice(d0, d0+ch), slice(h0, h0+cw), slice(w0, w0+cd))

    # SNR matching (exact formula from mix_patches_augmentation.py):
    #   scale = sqrt(noise_canvas_region / noise_source_bg) * rand_snr_weight
    #   noise_canvas_region = E[canvas_region[crop_lbl==0]²]
    #   noise_source_bg     = E[crop_img[crop_lbl==0]²]
    canvas_region       = aug_img[sl]
    noise_canvas_region = float(np.mean(canvas_region[crop_lbl == 0].astype(np.float64) ** 2))
    noise_source_bg     = float(np.mean(crop_img[crop_lbl == 0].astype(np.float64) ** 2))
    rand_snr_weight     = rng.uniform(*rand_weight_range)
    scale               = float(np.sqrt(noise_canvas_region / (noise_source_bg + 1e-12))) * rand_snr_weight

    print(f"  noise_canvas={noise_canvas_region:.6f}  noise_src_bg={noise_source_bg:.6f}  "
          f"rand_snr_weight={rand_snr_weight:.3f}  scale={scale:.4f}")

    scaled_crop = crop_img.copy().astype(np.float32)
    scaled_crop[fg_mask] = np.clip(
        scaled_crop[fg_mask] * scale,
        scaled_crop.min(), scaled_crop.max()
    )

    dest_img_p = aug_img[sl].copy()
    dest_lbl_p = aug_lbl[sl].copy()
    dest_img_p[fg_mask] = scaled_crop[fg_mask]
    dest_lbl_p[fg_mask] = crop_lbl[fg_mask]
    aug_img[sl] = dest_img_p
    aug_lbl[sl] = dest_lbl_p

    snr_source = compute_snr(crop_img, crop_lbl)
    snr_after  = compute_snr(aug_img, aug_lbl)

    print(f"  SNR: source={snr_source:.4f}  canvas={compute_snr(canvas_img, canvas_lbl):.4f}  "
          f"after={snr_after:.4f}")

    info = {
        'insert_loc':      (d0, h0, w0),
        'snr_source':      round(snr_source, 4),
        'snr_canvas':      round(compute_snr(canvas_img, canvas_lbl), 4),
        'rand_snr_weight': round(rand_snr_weight, 3),
        'scale':           round(scale, 4),
        'snr_after':       round(snr_after, 4),
        'fg_inserted':     int(fg_mask.sum()),
    }
    return aug_img, aug_lbl, info


# ─────────────────────────────────────────────────────────────────────────────
# Main augmentation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate_augmented_patches(empty_patches, source_patches,
                                total_voxels, fg_target, max_patches,
                                min_crop_size, dilation_iters, output_dir, rng,
                                num_classes):
    """
    Generate augmented patches using a round-robin loop across all active classes.
    One patch is generated per class per cycle, so the budget is distributed evenly.

    Stopping criterion — dedicated fg tracking:
      dedicated_running_fg[c] = original_fg[c]  +  sum of (aug_lbl == c).sum()
                                  from ALL patches where class_id == c only.

    Each class stops when dedicated_running_fg[c] >= fg_target.  Incidental fg
    from other classes' patches is NOT counted here, preventing dominant classes
    from reaching their target prematurely via cross-class leakage.

    The global loop stops when all classes are done or max_patches is hit.

    Returns (augmented_entries, inserted_fg_per_class).
    """
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    augmented_entries = []
    inserted_fg           = {c: 0 for c in range(1, num_classes)}
    running_fg            = {c: int(total_voxels[c]) for c in range(num_classes)}
    dedicated_running_fg  = {c: int(total_voxels[c]) for c in range(1, num_classes)}
    aug_idx     = 0
    total_generated = 0

    # Initialise per-class state for active classes
    per_class = {}
    print(f"\n{'='*60}")
    for c in range(1, num_classes):
        srcs = source_patches.get(c, [])
        if not srcs:
            print(f"  Class {c}: no source patches — skipping")
            continue
        if dedicated_running_fg[c] >= fg_target:
            print(f"  Class {c}: already at target — skipping")
            continue
        per_class[c] = {'srcs': srcs, 'generated': 0, 'attempts': 0,
                        'snr_log': [], 'skip_counts': {}}
        print(f"  Class {c}: deficit={fg_target - running_fg[c]:,}  sources={len(srcs)}")
    print(f"  Canvas patches : {len(empty_patches)}")
    print(f"  Global budget  : {max_patches} patches")
    print(f"{'='*60}")

    active = list(per_class.keys())
    if not active:
        print("  No classes need augmentation.")
        return augmented_entries, inserted_fg

    pbar = tqdm.tqdm(total=max_patches, desc="  augmenting", unit="patch")

    # Round-robin: one patch per class per cycle
    while active and total_generated < max_patches:
        for class_id in list(active):

            # Check if this class has reached its target (dedicated fg only)
            if dedicated_running_fg[class_id] >= fg_target:
                tqdm.tqdm.write(
                    f"  ✓ Class {class_id} reached target  "
                    f"({per_class[class_id]['generated']} patches, "
                    f"dedicated_fg={dedicated_running_fg[class_id]:,})"
                )
                active.remove(class_id)
                continue

            if total_generated >= max_patches:
                break

            state = per_class[class_id]
            state['attempts'] += 1

            # Safety: avoid infinite retries for a single class slot
            if state['attempts'] > max_patches * 20:
                tqdm.tqdm.write(f"  [WARNING] Class {class_id}: too many failed attempts — removing")
                active.remove(class_id)
                break

            srcs   = state['srcs']
            src    = srcs[rng.randint(0, len(srcs))]
            canvas = empty_patches[rng.randint(0, len(empty_patches))]

            src_img    = load_patch(src['image'])
            src_lbl    = load_patch(src['label']).astype(np.int32)
            canvas_img = load_patch(canvas['image'])
            canvas_lbl = load_patch(canvas['label']).astype(np.int32)

            crop_img, crop_lbl, ok, crop_info = crop_around_class(
                src_img, src_lbl, class_id, min_crop_size, dilation_iters, rng
            )
            if not ok:
                reason = crop_info if isinstance(crop_info, str) else "crop failed"
                state['skip_counts'][reason] = state['skip_counts'].get(reason, 0) + 1
                continue

            aug_img, aug_lbl, ins_info = insert_with_snr_match(
                canvas_img, canvas_lbl, crop_img, crop_lbl, rng
            )
            if 'skipped' in ins_info:
                state['skip_counts'][ins_info['skipped']] = \
                    state['skip_counts'].get(ins_info['skipped'], 0) + 1
                continue

            # Update running_fg for ALL classes (for logging/display)
            aug_cls_counts = {c: int((aug_lbl == c).sum()) for c in range(num_classes)}
            for c in range(1, num_classes):
                running_fg[c] += aug_cls_counts[c]
            # Update dedicated_running_fg only for the targeted class
            dedicated_running_fg[class_id] += aug_cls_counts[class_id]
            inserted_fg[class_id] += ins_info['fg_inserted']

            # Per-patch log — show dedicated fg (stopping criterion) vs total
            fg_status = "  ".join(
                f"c{c}={dedicated_running_fg[c]:,}/{fg_target:,}" for c in per_class
            )
            tqdm.tqdm.write(
                f"  [c{class_id}] patch {state['generated']+1}  "
                f"src={Path(src['image']).name}  canvas={Path(canvas['image']).name}\n"
                f"    crop: start_coord={crop_info['start_coord']}  "
                f"fg_in_crop={crop_info['fg_in_crop']}  "
                f"classes_in_crop={crop_info['classes_in_crop']}\n"
                f"    insert@{ins_info['insert_loc']}  scale={ins_info['scale']}  "
                f"snr_after={ins_info['snr_after']}\n"
                f"    aug_counts={aug_cls_counts}\n"
                f"    running_fg: {fg_status}"
            )

            state['snr_log'].append(ins_info['snr_after'])

            img_path = os.path.join(img_dir, f"aug_c{class_id}_{aug_idx:05d}.pt")
            lbl_path = os.path.join(lbl_dir, f"aug_c{class_id}_{aug_idx:05d}.pt")
            save_patch(aug_img.astype(np.float32), img_path)
            save_patch(aug_lbl.astype(np.float32), lbl_path)

            augmented_entries.append({'image': img_path, 'label': lbl_path})
            aug_idx         += 1
            state['generated'] += 1
            total_generated    += 1
            pbar.update(1)

    pbar.close()

    # Per-class summary
    print()
    for c, state in per_class.items():
        reached = dedicated_running_fg[c] >= fg_target
        print(f"  ── Class {c} summary ──")
        print(f"  Generated     : {state['generated']}  ({state['attempts']} attempts)")
        print(f"  dedicated_fg  : {dedicated_running_fg[c]:,} / {fg_target:,}  "
              f"({'REACHED' if reached else 'NOT REACHED — hit max_patches cap'})")
        print(f"  running_fg    : {running_fg[c]:,}  (incl. incidental fg from all patches)")
        if not reached:
            print(f"  [WARNING] Increase --max-patches to reach the fg target.")
        if state['skip_counts']:
            print(f"  Skips     : {state['skip_counts']}")
        if state['snr_log']:
            print(f"  SNR after : min={min(state['snr_log']):.4f}  "
                  f"max={max(state['snr_log']):.4f}  "
                  f"mean={np.mean(state['snr_log']):.4f}")
        print()

    return augmented_entries, inserted_fg


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="Class-balanced patch augmentation for CryoDINO fine-tuning"
    )
    parser.add_argument("--datalist",           required=True,             help="Input datalist JSON")
    parser.add_argument("--output-dir",         required=True,             help="Output directory")
    parser.add_argument("--num-classes",        type=int,   default=4,     help="Total classes incl. background (default: 4)")
    parser.add_argument("--min-crop-size",       type=int,   default=128,   help="Minimum crop size per axis; max is the full patch dim (default: 128)")
    parser.add_argument("--target-multiplier",  type=float, default=2.0,   help="Universal fg target = V_max × multiplier (default: 2.0)")
    parser.add_argument("--max-patches",        type=int,   default=500,   help="Global cap on total augmented patches across all classes (default: 500)")
    parser.add_argument("--empty-threshold",    type=float, default=0.02,  help="Max fg fraction for a canvas patch (default: 0.02)")
    parser.add_argument("--source-threshold",   type=float, default=0.001, help="Min class fraction to qualify as source (default: 0.001)")
    parser.add_argument("--dilation-iters",     type=int,   default=3,     help="Dilation iterations around foreground (default: 3)")
    parser.add_argument("--seed",               type=int,   default=42)
    return parser.parse_args()


def main():
    args = get_args()
    rng  = np.random.RandomState(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CryoDINO Mix-Patch Augmentation")
    print(f"  datalist        : {args.datalist}")
    print(f"  output-dir      : {args.output_dir}")
    print(f"  num-classes     : {args.num_classes}")
    print(f"  min-crop-size   : {args.min_crop_size} (max = full patch dim per axis)")
    print(f"  target-mult     : {args.target_multiplier}")
    print(f"  max-patches     : {args.max_patches}")
    print(f"  empty-threshold : {args.empty_threshold}")
    print(f"  source-threshold: {args.source_threshold}")
    print(f"  dilation-iters  : {args.dilation_iters}")
    print(f"  seed            : {args.seed}")
    print(f"{'='*60}")

    with open(args.datalist) as f:
        datalist = json.load(f)

    train_entries = datalist.get("training", [])
    n_train = len(train_entries)
    print(f"\nLoaded: {n_train} training | "
          f"{len(datalist.get('validation', []))} val | "
          f"{len(datalist.get('test', []))} test")

    # Step 1: analyse
    stats, total_voxels = analyse_patches(train_entries, args.num_classes)

    # Step 2: split into canvases and per-class sources
    empty_patches, source_patches = split_patches(
        stats, args.empty_threshold, args.source_threshold, args.num_classes
    )

    if not empty_patches:
        print("\n[ERROR] No empty canvas patches found. Try increasing --empty-threshold.")
        return

    # Step 3: print plan and compute universal fg target
    fg_target = print_augmentation_plan(total_voxels, args.num_classes, args.target_multiplier)

    # Step 4: generate — loop per class until fg_target met or max_patches hit
    augmented_entries, inserted_fg = generate_augmented_patches(
        empty_patches=empty_patches,
        source_patches=source_patches,
        total_voxels=total_voxels,
        fg_target=fg_target,
        max_patches=args.max_patches,
        min_crop_size=args.min_crop_size,
        dilation_iters=args.dilation_iters,
        output_dir=args.output_dir,
        rng=rng,
        num_classes=args.num_classes,
    )

    # Step 5: re-analyse the full combined dataset to verify balancing
    print(f"\n{'='*60}")
    print(f"  Post-augmentation balance check")
    print(f"{'='*60}")
    all_entries = train_entries + augmented_entries
    _, total_voxels_after = analyse_patches(all_entries, args.num_classes)

    print(f"\n── Before vs After (foreground voxels per class) ──")
    for c in range(1, args.num_classes):
        before     = int(total_voxels[c])
        after      = int(total_voxels_after[c])
        ratio      = after / before if before > 0 else float('inf')
        net_new    = inserted_fg.get(c, 0)
        print(f"  Class {c}: {before:>12,}  →  {after:>12,}  (×{ratio:.2f})  "
              f"|  net new inserted: {net_new:,}")

    # Step 6: save updated datalist
    updated  = copy.deepcopy(datalist)
    updated["training"] = all_entries
    out_json = os.path.join(args.output_dir, Path(args.datalist).stem + "_augmented.json")
    with open(out_json, 'w') as f:
        json.dump(updated, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Original training  : {n_train}")
    print(f"  Augmented patches  : {len(augmented_entries)}")
    print(f"  Total training     : {len(all_entries)}")
    print(f"  Updated datalist   : {out_json}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
