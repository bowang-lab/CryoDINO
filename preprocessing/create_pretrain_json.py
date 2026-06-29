import os
import json
import argparse
import nibabel as nib
import numpy as np
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Create JSON file for pretraining from multiple folders.")
    parser.add_argument('--input_folders', type=str, nargs='+',
                        help='List of input folders containing tomogram files.')
    parser.add_argument('--base_path', type=str, default='',
                        help='Base path to prepend to each file path (default: current directory).')
    parser.add_argument('--output_json', type=str, default='pretrain_data.json',
                        help='Output JSON filename (default: pretrain_data.json)')
    parser.add_argument('--update-existing', type=str, default=None, metavar='EXISTING_JSON',
                        help='Read patch paths from an existing JSON and add tomo_mean/tomo_std. '
                             'Skips directory walk entirely — much faster than re-scanning.')
    return parser.parse_args()

_nifti_cache = {}

def pt_path_to_nifti(subtomogram_path):
    nifti_path = subtomogram_path.replace('_subtomograms', '')
    nifti_path = re.sub(r'_patch_.*\.pt$', '.nii.gz', nifti_path)
    return nifti_path

def get_nifti_metadata(subtomogram_path):
    """
    Converts .pt path → .nii.gz path and returns (spacing, tomo_mean, tomo_std).
    Caches per tomogram so each NIfTI is loaded only once.
    """
    nifti_path = pt_path_to_nifti(subtomogram_path)

    if nifti_path in _nifti_cache:
        return _nifti_cache[nifti_path]

    if not os.path.exists(nifti_path):
        print(f"Warning: NIfTI not found: {nifti_path}")
        return None

    print(f"Loading: {nifti_path}")
    img = nib.load(nifti_path)
    spacing = [float(s) for s in img.header.get_zooms()]
    data = np.asarray(img.dataobj, dtype=np.float32)
    tomo_mean = float(data.mean())
    tomo_std = float(data.std())
    result = (spacing, tomo_mean, tomo_std)
    _nifti_cache[nifti_path] = result
    return result

def update_existing(existing_json, output_json):
    """Add tomo_mean/tomo_std to an existing pretrain JSON without re-walking directories."""
    with open(existing_json) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} entries from {existing_json}")

    updated, skipped = 0, 0
    for entry in dataset:
        meta = get_nifti_metadata(entry['image'])
        if meta:
            _, tomo_mean, tomo_std = meta
            entry['tomo_mean'] = tomo_mean
            entry['tomo_std'] = tomo_std
            updated += 1
        else:
            skipped += 1

    print(f"Updated {updated} entries, skipped {skipped}")
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"Saved to {output_json}")

def main():
    args = parse_args()

    if args.update_existing:
        update_existing(args.update_existing, args.output_json)
        return

    if not args.input_folders:
        raise ValueError("Provide --input_folders or --update-existing")

    dataset = []
    for folder in args.input_folders:
        folder_path = os.path.join(args.base_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist. Skipping.")
            continue

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.pt'):
                    full_path = os.path.join(root, file)
                    meta = get_nifti_metadata(full_path)
                    if meta:
                        spacing, tomo_mean, tomo_std = meta
                        dataset.append({
                            "image": full_path,
                            "shape": [128, 128, 128],
                            "spacing": spacing,
                            "tomo_mean": tomo_mean,
                            "tomo_std": tomo_std,
                        })

    with open(args.output_json, 'w') as json_file:
        json.dump(dataset, json_file, indent=4)
    print(f"Successfully created {args.output_json} with {len(dataset)} entries.")

if __name__ == "__main__":
    main()