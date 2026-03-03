import os 
import json
import argparse
import nibabel as nib
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Create JSON file for pretraining from multiple folders.")
    parser.add_argument('--input_folders', type=str, nargs='+', required=True,
                        help='List of input folders containing tomogram files.')
    parser.add_argument('--base_path', type=str, default='',
                        help='Base path to prepend to each file path (default: current directory).')
    parser.add_argument('--output_json', type=str, default='pretrain_data.json',
                        help='Output JSON filename (default: pretrain_data.json)')
    return parser.parse_args()

def get_nifti_metadata(subtomogram_path):
    """
    Uses regex to convert the .pt path to the .nii.gz path and extracts spacing and shape.
    """
    nifti_path = subtomogram_path.replace('_subtomograms', '')

    # This regex replaces '_patch_', followed by anything, ending with '.pt' 
    # with '.nii.gz'
    nifti_path = re.sub(r'_patch_.*\.pt$', '.nii.gz', nifti_path)
    
    if not os.path.exists(nifti_path):
        print(f"Warning: NIfTI file not found for {subtomogram_path} at {nifti_path}")
        return None

    img = nib.load(nifti_path)
    spacing = [float(s) for s in img.header.get_zooms()]    
    return spacing

def main():
    args = parse_args()
    
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
                    
                    spacing = get_nifti_metadata(full_path)
                    
                    if spacing:
                        dataset.append({
                            "image": full_path,
                            "shape": [128, 128, 128],
                            "spacing": spacing
                        })
    
    # Save as a list of dictionaries
    with open(args.output_json, 'w') as json_file:
        json.dump(dataset, json_file, indent=4)
    
    print(f"Successfully created {args.output_json} with {len(dataset)} entries.")

if __name__ == "__main__":
    main()