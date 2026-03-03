"""
This code is written by Ahmadreza Attarpour, attarpour1993@gmail.com
Adapted from subtomograms_generation.py for downstream segmentation datasets.

Loads image+label NIfTI pairs from a JSON datalist, creates 128^3 patches,
filters for patches with >= fg_threshold foreground voxels, saves as .pt files,
and generates a new JSON datalist.

Usage:
    # Percentile normalization only (default, 0.5-99.5 to [-1, 1])
    python downstream_patch_generation.py --datalist-json data.json --output-dir /path/to/patches

    # Z-score normalization only
    python downstream_patch_generation.py --datalist-json data.json --output-dir /path/to/patches --zscore

    # Both: z-score first, then percentile
    python downstream_patch_generation.py --datalist-json data.json --output-dir /path/to/patches --zscore --percentile

    # Custom patch size and foreground threshold
    python downstream_patch_generation.py --datalist-json data.json --output-dir /path/to/patches --patch-size 64 --fg-threshold 0.005

    # Different patch size for Z (depth) axis
    python downstream_patch_generation.py --datalist-json data.json --output-dir /path/to/patches --patch-size 512 --patch-size-z 128 --zscore
"""

import argparse
import json
import os
import numpy as np
import nibabel as nib
import torch
import einops
from monai.transforms import ScaleIntensityRangePercentiles


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
    parser = argparse.ArgumentParser(description='Generate 128^3 patches from downstream segmentation datasets')
    parser.add_argument('--datalist-json', required=True, help='Path to the datalist JSON file')
    parser.add_argument('--output-dir', required=True, help='Output directory for patches')
    parser.add_argument('--patch-size', type=int, default=128, help='Patch size for X and Y axes (default: 128)')
    parser.add_argument('--patch-size-z', type=int, default=None, help='Patch size for Z (depth) axis (default: same as --patch-size)')
    parser.add_argument('--fg-threshold', type=float, default=0.01, help='Min fraction of foreground voxels to keep a patch (default: 0.01)')
    parser.add_argument('--output-json', default=None, help='Output JSON datalist path (default: auto-generated)')
    parser.add_argument('--zscore', action='store_true', help='Apply nnUNet-style z-score normalization (mean=0, std=1)')
    parser.add_argument('--percentile', action='store_true', help='Apply ScaleIntensityRangePercentiles (0.5-99.5 to [-1, 1])')
    args = parser.parse_args()

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
