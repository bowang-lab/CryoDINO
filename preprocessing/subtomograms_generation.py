"""
This code is written by Ahmadreza Attarpour, attarpour1993@gmail.com
This code is written for preprocessing training image patches for CryoET foundation model project
IT loads a 3D nifti image, performs Otsu thresholding and creates patch_size^3 subtomograms and saves them in the output directory
"""
# import libraries
import pathlib
import numpy as np
import nibabel as nib
import argparse
import einops
import os 
from skimage import filters
import torch 

# Create the parser
my_parser = argparse.ArgumentParser(description='The directory of data')

# Add the arguments
my_parser.add_argument('-i','--input', help='input 3D nifti files directory', required=True)
my_parser.add_argument('-m','--mask', help='mask 3D nifti files directory', required=False, default=None)
my_parser.add_argument('-p','--patch_size', help='the outputs will be patch size x patch size x patch size', required=False, default=128, type=int)
my_parser.add_argument('-o','--output_path', help='output nifti files directory', required=True)


# image to patch function
def img_to_patch(tomogram: np.ndarray, patch_size: tuple):
        """
        Docstring for img_to_patch
        """
        W_new, H_new, D_new = patch_size
        W, H, D= tomogram.shape # Fixed: changed 'input.shape' to 'tomogram.shape'
        
        # zero pad if the input is not divisible by patch_size
        if H % H_new != 0:
            H = H_new * ((H // H_new) + 1)
        
        if W % W_new != 0:
            W = W_new * ((W // W_new) + 1)
        
        if D % D_new != 0:
            D = D_new * ((D // D_new) + 1)

        tomogram_rearranged = np.zeros((W, H, D))

        # pad the image with zeros
        tomogram_rearranged[:tomogram.shape[0], :tomogram.shape[1], :tomogram.shape[2]] = tomogram

        # pad the image with repeating the image
        tomogram_rearranged[tomogram.shape[0]:, tomogram.shape[1]:, tomogram.shape[2]:] = tomogram[:W-tomogram.shape[0], :H-tomogram.shape[1], :D-tomogram.shape[2]]

        temp1, temp2, temp3 = W // patch_size[0], H // patch_size[1], D // patch_size[2]

        tomogram_rearranged = einops.rearrange(tomogram_rearranged, '(b1 w) (b2 h) (b3 d) -> (b1 b2 b3) w h d', b1 = temp1,  b2=temp2, b3=temp3)

        # UPDATED RETURN: Returning the grid dimensions (temp1, temp2, temp3) alongside the patches
        return tomogram_rearranged, (temp1, temp2, temp3)


def otsu_threshold(tomogram: np.ndarray) -> np.ndarray:

    # Normalize the tomogram
    tomogram = tomogram.astype(np.float32)
    mean = tomogram.mean()
    std = tomogram.std()
    if std == 0:
        print(f'Warning: std is zero!')
        return np.zeros(tomogram.shape, dtype=bool)

    tomogram -= mean
    tomogram /= (max(std, 1e-8))

    # Compute Otsu threshold on the whole volume
    # Added check to ensure we have negative pixels before filtering, otherwise default to full image
    valid_pixels = tomogram[tomogram < 0]
    if valid_pixels.size > 0:
        thresh = filters.threshold_otsu(valid_pixels)
    else:
        thresh = filters.threshold_otsu(tomogram)
        
    mask = tomogram <= thresh

    return mask

def main(args):
     
    input_path = args['input']
    mask_path = args['mask']
    output_dir_img = args['output_path']
    
    # Threshold count: 16^3
    voxel_threshold = 16**3

    if not os.path.exists(output_dir_img): 
        os.makedirs(output_dir_img)  
    
    patch_size_val = args['patch_size']
    patch_size = (patch_size_val, patch_size_val, patch_size_val)

    # find the 3D nifti or tif inside the input_path
    file_names = pathlib.Path(input_path)
    img_list_name = [file for file in file_names.glob('*') if file.name.endswith('.nii') or file.name.endswith('.nii.gz')]

    mask_names = pathlib.Path(mask_path) if mask_path is not None else None
    if mask_path is not None:
        mask_list_name = [file for file in mask_names.glob('*') if file.name.endswith('.nii') or file.name.endswith('.nii.gz')]

    for img_path in img_list_name:
        img_name = img_path.name
        nii_obj = nib.load(str(img_path))

        # check if the image is valid
        try:
            image = nii_obj.get_fdata()

            original_affine = nii_obj.affine
            original_header = nii_obj.header
            
            print(f"Processing {img_name} - size: {image.shape}")
            # Check for NaN values
            NAN_SKIP_THRESHOLD = 0.01  # 1% threshold
            EXTREME_VALUE_THRESHOLD = 1e9  # Arbitrary large value threshold
            if np.isnan(image).any():
                nan_count = np.isnan(image).sum()
                total_voxels = image.size
                nan_ratio = nan_count / total_voxels
                
                if nan_ratio > NAN_SKIP_THRESHOLD:
                    print(f"  [ERROR] Skipping {img_name}: {nan_ratio:.2%} of voxels are NaN (Threshold: {NAN_SKIP_THRESHOLD:.0%}).")
                    continue # Skip to the next image
                else:
                    print(f"  [WARNING] Found {nan_count} NaN voxels ({nan_ratio:.4%}). Replacing with 0.0.")
                    # Fix the image in place so Otsu and Patching work safely
                    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

                max_val = np.max(np.abs(image))
                if max_val > EXTREME_VALUE_THRESHOLD:
                    print(f"  [SKIPPING] {img_name} has extreme values (Max: {max_val:.2e}). Threshold is {EXTREME_VALUE_THRESHOLD:.0e}.")
                    continue

            # 1. Compute/Load Mask on the FULL image first
            mask = None
            
            if mask_path is not None:
                # Construct the expected mask filename
                # Note: This assumes your input files end in _0000.nii.gz per your snippet
                # Check if the image is the 'filtered' version or the standard version to determine mask name
                if '_filtered_0000.nii.gz' in img_name:
                    target_mask_name = img_name.replace('_filtered_0000.nii.gz', '_denoised_deconv_lowpass_0000.nii.gz')
                else:
                    target_mask_name = img_name.replace('_0000.nii.gz', '_deconv_lowpass_0000.nii.gz')
                target_mask_full_path = os.path.join(mask_path, target_mask_name)
                
                if os.path.exists(target_mask_full_path):
                    print(f"  Loading mask from: {target_mask_name}")
                    mask = nib.load(target_mask_full_path).get_fdata()
                else:
                    print(f"  Warning: Mask {target_mask_name} not found. Falling back to Otsu.")

            # If mask is still None (either mask_path was None, or file didn't exist)
            if mask is None:
                print(f"  Computing Otsu threshold...")
                mask = otsu_threshold(image)

            # Save the full mask for reference
            # base_name = img_name.replace('.nii.gz','').replace('.nii', '').replace('_mask', '')
            # mask_name = f"{base_name}_otsu_mask.nii.gz"
            # nib.save(nib.Nifti1Image(mask.astype(np.uint8), original_affine, original_header), os.path.join(output_dir_img, mask_name))

            # 2. Patch both Image and Mask
            # We capture 'grid_dims' here to help with naming and affine calculation later
            img_patches, grid_dims = img_to_patch(image, patch_size)
            mask_patches, _ = img_to_patch(mask.astype(np.uint8), patch_size)
            
            print(f"  Patches created: {img_patches.shape}. Grid dimensions: {grid_dims}")

            b1, b2, b3 = grid_dims # These correspond to X, Y, Z blocks
            
            saved_count = 0
            
            # 3. Iterate through patches
            for i in range(img_patches.shape[0]):
                
                # Check for foreground content
                if np.sum(mask_patches[i]) > voxel_threshold:
                                    
                    # We need to unravel the flat index 'i' back to 3D grid coordinates (gx, gy, gz)
                    # Einops rearranges as (b1 b2 b3), so b3 is the fastest changing index
                    gz = i % b3
                    gy = (i // b3) % b2
                    gx = (i // (b3 * b2)) % b1
                    
                    # Calculate the pixel offset in the original large image
                    x_start = gx * patch_size_val
                    y_start = gy * patch_size_val
                    z_start = gz * patch_size_val
                    
                    # Calculate the NEW ORIGIN in physical space using the original affine
                    # Physical_Pos = Affine @ [pixel_x, pixel_y, pixel_z, 1]
                    old_origin_vector = np.array([x_start, y_start, z_start, 1])
                    new_origin_coords = original_affine.dot(old_origin_vector)
                    
                    # Create a new affine matrix for this specific patch
                    new_affine = original_affine.copy()
                    new_affine[:3, 3] = new_origin_coords[:3]
                                    
                    # Construct name: originalname_x_y_z.nii.gz
                    base_name = img_name.replace('.nii.gz', '').replace('.nii', '')
                    patch_name = f"{base_name}_patch_{x_start}_{y_start}_{z_start}.nii.gz"
                    
                    save_path = os.path.join(output_dir_img, patch_name)
                

                    # --- Nifti Strategy (Commented Out) ---
                    # Create Nifti object with the NEW affine
                    # patch_nii = nib.Nifti1Image(img_patches[i], new_affine, original_header)
                    # nib.save(patch_nii, save_path.replace('.pt', '.nii.gz'))
                    
                    # --- Torch Strategy ---
                    # Save tensor directly
                    torch.save(torch.from_numpy(img_patches[i]), save_path.replace('.nii.gz', '.pt'))                
                    saved_count += 1

            print(f"  Saved {saved_count} patches from {img_name}")

        except Exception as e:
            # --- THE SAFETY NET ---
            print(f"  [SKIPPING] Could not process {img_name}.")
            print(f"  Reason: {e}")
            continue

if __name__ == '__main__':
     
    # Execute the parse_args() method
    args = vars(my_parser.parse_args())
    main(args)