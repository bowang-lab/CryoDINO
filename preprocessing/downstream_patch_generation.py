"""
This code is written by Ahmadreza Attarpour, attarpour1993@gmail.com
Adapted from subtomograms_generation.py for downstream segmentation datasets.

Loads image+label NIfTI pairs from a JSON datalist, creates 128^3 patches,
filters for patches with >= fg_threshold foreground voxels, saves as .pt files,
and generates a new JSON datalist.

Detection mode (--detection --csv annotations.csv):
    Instead of image+label pairs, loads image NIfTI + CSV point annotations.
    Uses overlapping sliding-window patches (stride = patch_size // 2).
    For each patch: filters GT centers within bounds, converts to patch-local
    XYZ coords (subtract patch offset), saves as "points" key.
    Val/test entries keep original NIfTI paths intact with global GT points.

    CSV format: run, particle_name, z, y, x, voxel_size  (z/y/x in voxels)
    Coordinate convention (nibabel XYZ, matching pretraining):
      NIfTI axis 0 = X (~630 vox), axis 2 = Z (thin ~184 vox)
      Points stored as (x_vox, y_vox, z_vox, class_id, sigma_vox)

Usage:
    # Segmentation — percentile normalization only (default, 0.5-99.5 to [-1, 1])
    python downstream_patch_generation.py --datalist-json data.json --output-dir /path/to/patches

    # Segmentation — z-score normalization only
    python downstream_patch_generation.py --datalist-json data.json --output-dir /path/to/patches --zscore

    # Segmentation — both: z-score first, then percentile
    python downstream_patch_generation.py --datalist-json data.json --output-dir /path/to/patches --zscore --percentile

    # Segmentation — custom patch size and foreground threshold
    python downstream_patch_generation.py --datalist-json data.json --output-dir /path/to/patches --patch-size 64 --fg-threshold 0.005

    # Segmentation — different patch size for Z (depth) axis
    python downstream_patch_generation.py --datalist-json data.json --output-dir /path/to/patches --patch-size 512 --patch-size-z 128 --zscore

    # Detection — CZI dataset
    python downstream_patch_generation.py \\
        --datalist-json /path/to/czi_split_datalist.json \\
        --csv           dataset/downstream_detection/Dataset440_CZII_10440/point_annotations.csv \\
        --output-dir    /path/to/detection_patches \\
        --output-json   /path/to/czi_100_datalist.json \\
        --detection --zscore --patch-size 128 \\
        --sigmas-ang '{"Beta-amylase":65,"Beta-galactosidase":90,"Thyroglobulin":130,"cytosolic ribosome":150,"ferritin complex":60,"virus-like capsid":135}'
"""

import argparse
import csv
import json
import os
import numpy as np
import nibabel as nib
import torch
import einops
from monai.transforms import ScaleIntensityRangePercentiles


# ---------------------------------------------------------------------------
# Detection helpers — ported from Kaggle functional.py / mixin.py
# ---------------------------------------------------------------------------

def get_points_mask_within_patch(points: np.ndarray, patch_shape: tuple) -> np.ndarray:
    """Boolean mask for points inside the patch volume.

    Ported from Kaggle get_points_mask_within_cube (functional.py).
    Adapted for nibabel XYZ convention: points[:,0]=X, [:,1]=Y, [:,2]=Z;
    patch_shape = (X, Y, Z) unlike Kaggle's (D, H, W) = (Z, Y, X).

    Args:
        points      : (N, 3+) array, first 3 cols are (x, y, z) in voxels
        patch_shape : (X_dim, Y_dim, Z_dim) of the patch
    """
    return (
        (points[:, 0] >= 0) & (points[:, 0] < patch_shape[0])
        & (points[:, 1] >= 0) & (points[:, 1] < patch_shape[1])
        & (points[:, 2] >= 0) & (points[:, 2] < patch_shape[2])
    )


def detection_tile_positions(dim: int, patch_size: int, stride: int) -> list:
    """Overlapping tile start positions along one axis.

    Every tile is exactly patch_size wide. The last position is snapped to
    dim - patch_size so tiles always fit without padding.
    Mirrors tile_positions() in detection3d.py sliding_window_accumulate.
    """
    if dim <= patch_size:
        return [0]
    stops = list(range(0, dim - patch_size + 1, stride))
    if not stops or stops[-1] + patch_size < dim:
        stops.append(dim - patch_size)
    return stops


def run_name_from_nii(nii_path: str) -> str:
    """'TS_5_4_0000.nii.gz' → 'TS_5_4'  (strips nnU-Net _XXXX channel suffix)."""
    base = os.path.basename(nii_path).replace('.nii.gz', '').replace('.nii', '')
    if '_' in base and base[-5] == '_' and base[-4:].isdigit():
        base = base[:-5]
    return base


def load_detection_annotations(csv_path: str, sigmas_ang: dict, default_sigma_ang: float) -> tuple:
    """Load point annotation CSV; return (annotations, class_map).

    CSV format: run, particle_name, z, y, x, voxel_size  (z/y/x in voxels).
    Coordinate reorder — CSV (z, y, x) → stored as (x_vox, y_vox, z_vox),
    matching nibabel XYZ convention and augmentations.py points format.

    Class IDs are auto-assigned from sorted unique particle names (0-indexed,
    alphabetical) — no dataset-specific hardcoding; works for CZI, BYU, etc.
    Matches Kaggle mixin.py labels tensor column order: (x, y, z, class_id, sigma).

    Args:
        csv_path          : path to annotations CSV
        sigmas_ang        : {particle_name: radius_in_angstroms} — used as sigma_vox
        default_sigma_ang : fallback sigma (Å) for particles not in sigmas_ang

    Returns:
        annotations : {run: {'points': [[x,y,z,cls,sigma],...], 'voxel_size': float}}
        class_map   : {particle_name: class_id}
    """
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    # Auto class ID: sorted unique particle names → 0-indexed
    particle_names = sorted({r['particle_name'].strip() for r in rows})
    class_map = {name: idx for idx, name in enumerate(particle_names)}
    print(f"  Auto class map: {class_map}")

    annotations = {}
    for row in rows:
        run      = row['run'].strip()
        particle = row['particle_name'].strip()
        z_vox    = float(row['z'])
        y_vox    = float(row['y'])
        x_vox    = float(row['x'])
        vs       = float(row['voxel_size'])

        # CSV (z, y, x) are already in voxels; reorder to XYZ (col 0=x, 1=y, 2=z)
        class_id  = float(class_map[particle])
        sigma_vox = sigmas_ang.get(particle, default_sigma_ang) / vs

        if run not in annotations:
            annotations[run] = {'points': [], 'voxel_size': vs}
        annotations[run]['points'].append([x_vox, y_vox, z_vox, class_id, sigma_vox])

    for run, info in annotations.items():
        print(f"  {run}: {len(info['points'])} particles  voxel_size={info['voxel_size']} Å/vox")

    return annotations, class_map


def main_detection(args):
    """Detection mode: overlapping patches for train, intact NIfTI for val/test.

    Reads train/val/test split from --datalist-json (same as segmentation flow).
    Each entry needs at least an "image" key pointing to a NIfTI file.
    GT annotations come from --csv; run name is derived from the NIfTI filename.

    For each training tomogram:
      1. Load NIfTI with nibabel → XYZ order (axis 0=X, axis 2=Z)
      2. Z-score normalize full volume
      3. Sliding-window overlapping patches (stride = patch_size // 2)
      4. Per patch: subtract patch offset → local XYZ coords, then filter with
         get_points_mask_within_patch  (Kaggle functional.py get_points_mask_within_cube)
      5. Save patch as .pt; record {"image", "points", "voxel_size"} entry

    Val/test: keep original NIfTI paths + global XYZ GT points for sliding-window metric.
    """
    patch_size = args.patch_size
    stride     = max(1, patch_size // 2)

    sigmas_ang        = json.loads(args.sigmas_ang) if args.sigmas_ang else {}
    default_sigma_ang = args.default_sigma_ang

    print("Detection mode")
    print(f"  Patch size : {patch_size}³  stride={stride}")
    print(f"  Sigmas (Å) : {sigmas_ang}  default={default_sigma_ang}")

    # Load annotations from CSV
    print("\nLoading annotations...")
    annotations, class_map = load_detection_annotations(args.csv, sigmas_ang, default_sigma_ang)

    # Load datalist JSON (provides train/val/test split — same flow as segmentation)
    with open(args.datalist_json, 'r') as f:
        datalist = json.load(f)
    print(f"\nDatalist: {len(datalist['training'])} train  "
          f"{len(datalist['validation'])} val  {len(datalist['test'])} test")

    img_out_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(img_out_dir, exist_ok=True)

    train_entries = []

    # ---- Training: overlapping sliding-window patches ----------------------
    print("\n=== Training tomograms ===")
    for entry in datalist['training']:
        nii_path = entry['image']
        run      = run_name_from_nii(nii_path)
        print(f"\n[{run}]  {nii_path}")

        # Load NIfTI with nibabel → XYZ: shape (X, Y, Z)
        volume = nib.load(nii_path).get_fdata().astype(np.float32)
        if args.zscore:
            volume = zscore_normalize(volume)
        X, Y, Z = volume.shape
        print(f"  shape={volume.shape}  min={volume.min():.2f} max={volume.max():.2f}")

        run_info = annotations.get(run, {'points': [], 'voxel_size': 10.0})
        vs       = float(run_info['voxel_size'])
        all_pts  = (np.array(run_info['points'], dtype=np.float32)
                    if run_info['points'] else np.zeros((0, 5), dtype=np.float32))

        x_starts = detection_tile_positions(X, patch_size, stride)
        y_starts = detection_tile_positions(Y, patch_size, stride)
        z_starts = detection_tile_positions(Z, patch_size, stride)
        n_total  = len(x_starts) * len(y_starts) * len(z_starts)
        print(f"  grid {len(x_starts)}×{len(y_starts)}×{len(z_starts)} = {n_total} patches")

        run_basename = run_name_from_nii(nii_path)
        n_particle = 0

        for x0 in x_starts:
            for y0 in y_starts:
                for z0 in z_starts:
                    patch = volume[x0:x0+patch_size, y0:y0+patch_size, z0:z0+patch_size].copy()

                    # Subtract patch offset → patch-local XYZ coords (Kaggle logic)
                    patch_points = []
                    if len(all_pts) > 0:
                        local_pts = all_pts.copy()
                        local_pts[:, 0] -= x0   # x
                        local_pts[:, 1] -= y0   # y
                        local_pts[:, 2] -= z0   # z

                        # Filter: keep only centers inside the patch (Kaggle get_points_mask_within_cube)
                        keep = get_points_mask_within_patch(local_pts, patch.shape)
                        for pt in local_pts[keep]:
                            patch_points.append(pt.tolist())

                    has_particles = len(patch_points) > 0
                    # Always include one padding row so "points" shape is always (N, 5)
                    if not has_particles:
                        patch_points = [[-100.0, -100.0, -100.0, -100.0, -100.0]]
                    else:
                        n_particle += 1

                    patch_name = f"{run_basename}_{x0}_{y0}_{z0}.pt"
                    patch_path = os.path.join(img_out_dir, patch_name)
                    torch.save(torch.from_numpy(patch), patch_path)

                    train_entries.append({
                        'image':      patch_path,
                        'points':     patch_points,   # [[x,y,z,cls,sigma],...] local coords
                        'voxel_size': vs,
                    })

        print(f"  particle patches: {n_particle}/{n_total}  background: {n_total - n_particle}")

    # ---- Val/test: original NIfTI paths + global GT points ----------------
    # Val/test entries are kept intact (no patching); detection_val_iter runs
    # sliding_window_accumulate on the full volume and needs global GT coords.
    val_entries  = []
    test_entries = []

    for split, split_list, entries in [('val', datalist['validation'], val_entries),
                                        ('test', datalist['test'],       test_entries)]:
        if not split_list:
            continue
        print(f"\n=== {split.capitalize()} tomograms ===")
        for entry in split_list:
            nii_path   = entry['image']
            run        = run_name_from_nii(nii_path)
            run_info   = annotations.get(run, {'points': [], 'voxel_size': 10.0})
            vs         = float(run_info['voxel_size'])
            all_points = run_info['points'] if run_info['points'] else [[-100.0, -100.0, -100.0, -100.0, -100.0]]
            print(f"  {run}: {len(run_info['points'])} particles  voxel_size={vs}")
            entries.append({
                'image':      nii_path,    # original NIfTI — loaded by val_transforms
                'points':     all_points,  # global XYZ voxel coords
                'voxel_size': vs,
            })

    # ---- Save datalist JSON -----------------------------------------------
    datalist = {'training': train_entries, 'validation': val_entries, 'test': test_entries}
    output_json = args.output_json or os.path.join(args.output_dir, 'czi_100_datalist.json')
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(datalist, f, indent=2)

    print(f"\nDatalist saved: {output_json}")
    print(f"  Training   : {len(train_entries)}")
    print(f"  Validation : {len(val_entries)}")
    print(f"  Test       : {len(test_entries)}")
    print(f"  Class map  : {class_map}")


def img_to_patch(tomogram: np.ndarray, patch_size: tuple):
    """
    Split a 3D volume into non-overlapping patches of given size.
    Zero-pads if the volume is not evenly divisible.

    Returns:
        patches: (N, pw, ph, pd) array of patches
        grid_dims: (b1, b2, b3) number of patches along each axis
    """
    W_new, H_new, D_new = patch_size
    W, H, D = tomogram.shape

    # zero pad if the input is not divisible by patch_size
    if H % H_new != 0:
        H = H_new * ((H // H_new) + 1)
    if W % W_new != 0:
        W = W_new * ((W // W_new) + 1)
    if D % D_new != 0:
        D = D_new * ((D // D_new) + 1)

    tomogram_rearranged = np.zeros((W, H, D), dtype=tomogram.dtype)
    tomogram_rearranged[:tomogram.shape[0], :tomogram.shape[1], :tomogram.shape[2]] = tomogram

    temp1, temp2, temp3 = W // patch_size[0], H // patch_size[1], D // patch_size[2]
    tomogram_rearranged = einops.rearrange(
        tomogram_rearranged,
        '(b1 w) (b2 h) (b3 d) -> (b1 b2 b3) w h d',
        b1=temp1, b2=temp2, b3=temp3
    )

    return tomogram_rearranged, (temp1, temp2, temp3)


def zscore_normalize(image: np.ndarray, seg: np.ndarray = None, use_mask_for_norm: bool = False) -> np.ndarray:
    """nnUNet-style z-score normalization (mean=0, std=1)."""
    image = image.astype(np.float32, copy=False)
    if use_mask_for_norm and seg is not None:
        mask = seg > 0
        mean = image[mask].mean()
        std = image[mask].std()
        image[mask] = (image[mask] - mean) / max(std, 1e-8)
    else:
        mean = image.mean()
        std = image.std()
        image -= mean
        image /= max(std, 1e-8)
    return image


def main():
    parser = argparse.ArgumentParser(description='Generate patches from downstream segmentation or detection datasets')
    # Shared args
    parser.add_argument('--datalist-json', required=True, help='Path to the datalist JSON file (image paths + train/val/test split)')
    parser.add_argument('--output-dir', required=True, help='Output directory for patches')
    parser.add_argument('--patch-size', type=int, default=128, help='Patch size (default: 128)')
    parser.add_argument('--output-json', default=None, help='Output JSON datalist path (default: auto-generated)')
    parser.add_argument('--zscore', action='store_true', help='Apply nnUNet-style z-score normalization (mean=0, std=1)')
    parser.add_argument('--percentile', action='store_true', help='Apply ScaleIntensityRangePercentiles (0.5-99.5 to [-1, 1])')
    # Segmentation-only args
    parser.add_argument('--patch-size-z', type=int, default=None, help='[Segmentation] Patch size for Z axis (default: same as --patch-size)')
    parser.add_argument('--fg-threshold', type=float, default=0.01, help='[Segmentation] Min foreground fraction to keep a patch (default: 0.01)')
    # Detection args
    parser.add_argument('--detection', action='store_true', help='Detection mode: patches with GT point coords instead of label masks')
    parser.add_argument('--csv', default=None, help='[Detection] Point annotations CSV (run,particle_name,z,y,x,voxel_size)')
    parser.add_argument('--sigmas-ang', default=None,
                        help='[Detection] JSON string mapping particle_name→radius_Å, e.g. \'{"ferritin complex": 60}\'')
    parser.add_argument('--default-sigma-ang', type=float, default=0.0,
                        help='[Detection] Fallback sigma in Å for particles not in --sigmas-ang (default: 0.0)')
    args = parser.parse_args()

    # Route to detection mode
    if args.detection:
        if not args.csv:
            parser.error('--detection requires --csv')
        main_detection(args)
        return

    patch_size_z = args.patch_size_z if args.patch_size_z is not None else args.patch_size

    norm_desc = []
    if args.zscore:
        norm_desc.append("zscore")
    if args.percentile:
        norm_desc.append("percentile")
    print(f"Normalization: {' -> '.join(norm_desc)}")
    print(f"Patch size: ({args.patch_size}, {args.patch_size}, {patch_size_z})")

    with open(args.datalist_json, 'r') as f:
        datalist = json.load(f)

    patch_size = (args.patch_size, args.patch_size, patch_size_z)
    img_output_dir = os.path.join(args.output_dir, 'images')
    lbl_output_dir = os.path.join(args.output_dir, 'labels')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(lbl_output_dir, exist_ok=True)

    new_training_entries = []
    total_patches = 0
    total_kept = 0

    for entry in datalist['training']:
        img_path = entry['image']
        lbl_path = entry['label']

        print(f"Processing: {os.path.basename(img_path)}")

        # Load NIfTI volumes
        image = nib.load(img_path).get_fdata().astype(np.float32)
        label = nib.load(lbl_path).get_fdata().astype(np.float32)

        print(f"  Image shape: {image.shape}, Label shape: {label.shape}")

        # Apply normalization on full volume before patching
        if args.zscore:
            image = zscore_normalize(image, seg=label, use_mask_for_norm=False)
        if args.percentile:
            normalizer = ScaleIntensityRangePercentiles(
                lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False
            )
            image = normalizer(image)
        print(f"  Normalized: min={image.min():.3f}, max={image.max():.3f}, mean={image.mean():.3f}, std={image.std():.3f}")

        # Create patches from both image and label using the same grid
        img_patches, grid_dims = img_to_patch(image, patch_size)
        lbl_patches, _ = img_to_patch(label, patch_size)

        b1, b2, b3 = grid_dims
        print(f"  Grid: {grid_dims}, Total patches: {img_patches.shape[0]}")

        # Derive base name from image filename
        img_basename = os.path.basename(img_path).replace('.nii.gz', '').replace('.nii', '')

        kept = 0
        for i in range(img_patches.shape[0]):
            lbl_patch = lbl_patches[i]
            fg_fraction = (lbl_patch > 0).sum() / lbl_patch.size

            if fg_fraction >= args.fg_threshold:
                # Compute grid coordinates
                gz = i % b3
                gy = (i // b3) % b2
                gx = (i // (b3 * b2)) % b1

                x_start = gx * args.patch_size
                y_start = gy * args.patch_size
                z_start = gz * patch_size_z

                patch_name = f"{img_basename}_patch_{x_start}_{y_start}_{z_start}.pt"

                img_save_path = os.path.join(img_output_dir, patch_name)
                lbl_save_path = os.path.join(lbl_output_dir, patch_name)

                torch.save(torch.from_numpy(img_patches[i].copy()), img_save_path)
                torch.save(torch.from_numpy(lbl_patches[i].copy()), lbl_save_path)

                new_training_entries.append({
                    'image': img_save_path,
                    'label': lbl_save_path,
                })
                kept += 1

        total_patches += img_patches.shape[0]
        total_kept += kept
        print(f"  Kept {kept}/{img_patches.shape[0]} patches (>= {args.fg_threshold:.0%} foreground)")

    print(f"\nTotal: kept {total_kept}/{total_patches} patches across all training volumes")

    # Build new datalist: patched training, original val/test
    new_datalist = {
        'training': new_training_entries,
        'validation': datalist['validation'],
        'test': datalist['test'],
    }

    # Save new datalist JSON
    if args.output_json:
        output_json_path = args.output_json
    else:
        input_json_dir = os.path.dirname(args.datalist_json)
        input_json_name = os.path.basename(args.datalist_json).replace('_100_datalist.json', '')
        output_json_path = os.path.join(input_json_dir, f"{input_json_name}_patches_100_datalist.json")

    with open(output_json_path, 'w') as f:
        json.dump(new_datalist, f, indent=2)

    print(f"New datalist saved to: {output_json_path}")
    print(f"  Training entries: {len(new_training_entries)}")
    print(f"  Validation entries: {len(datalist['validation'])}")
    print(f"  Test entries: {len(datalist['test'])}")


if __name__ == '__main__':
    main()
