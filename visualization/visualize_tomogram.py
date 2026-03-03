import argparse
import os
import nibabel as nib
import numpy as np
import matplotlib
# Use 'Agg' backend for cluster/headless compatibility
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Visualize NIfTI slices as a GIF or Mosaic.")
    parser.add_argument('--input_nifti', type=str, required=True, help='Path to the .nii or .nii.gz file')
    parser.add_argument('--output', type=str, default='output', help='Output filename (without extension)')
    parser.add_argument('--mode', type=str, choices=['gif', 'mosaic'], default='mosaic', 
                        help='Output type: "gif" for animation or "mosaic" for a 4x4 grid PNG')
    parser.add_argument('--step', type=int, default=10, help='Slice step size for GIF (default: 10)')    
    parser.add_argument('--down', type=int, default=4, help='Downsample x and y factor (default: 4)')    
    args = parser.parse_args()

    # 2. Check if file exists
    if not os.path.exists(args.input_nifti):
        print(f"Error: File {args.input_nifti} not found.")
        return

    print(f"Loading {args.input_nifti}...")
    img = nib.load(args.input_nifti)
    data = img.dataobj
    z_dim = data.shape[2]
    print(f"Image Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    full_data = np.asarray(data)
    print(f"Min: {full_data.min()}, Max: {full_data.max()}")

    # Determine if data is discrete (labels)
    is_discrete = np.issubdtype(full_data.dtype, np.integer) or len(np.unique(full_data)) <= 256

    if is_discrete:
        unique_vals = np.unique(full_data)
        cmap = plt.cm.get_cmap('tab20', len(unique_vals))
        imshow_kwargs = {'cmap': cmap, 'interpolation': 'nearest',
                         'vmin': unique_vals.min() - 0.5, 'vmax': unique_vals.max() + 0.5}
        print(f"Detected discrete data with {len(unique_vals)} unique values: {unique_vals}")
    else:
        imshow_kwargs = {'cmap': 'gray'}

    # 3. Handle Mosaic Mode
    if args.mode == 'mosaic':
        print("Generating 4x4 mosaic...")
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        indices = np.linspace(0, z_dim - 1, 16, dtype=int)
        
        for i, ax in enumerate(axes.flat):
            idx = indices[i]
            # Slicing: [x, y, z] -> Downsample x,y and select specific z
            slice_data = data[::args.down, ::args.down, idx].T
            ax.imshow(slice_data, origin='lower', **imshow_kwargs)
            ax.set_title(f"Slice {idx}")
            ax.axis('off')
        
        plt.tight_layout()
        out_file = f"{args.output}.png"
        plt.savefig(out_file)
        print(f"Success! Mosaic saved as: {out_file}")

    # 4. Handle GIF Mode
    else:
        fig, ax = plt.subplots()
        slices = range(0, z_dim, args.step)
        print(f"Generating animation for {len(slices)} slices...")

        def update(i):
            ax.clear()
            ax.imshow(data[::args.down, ::args.down, i].T, origin='lower', **imshow_kwargs)
            ax.set_title(f"Slice {i}")
            ax.axis('off')

        ani = animation.FuncAnimation(fig, update, frames=slices, interval=50)
        
        out_file = f"{args.output}.gif"
        try:
            # Attempt to use ffmpeg if available
            ani.save(out_file, writer='ffmpeg')
        except:
            print("ffmpeg not found, falling back to Pillow (this may be slow)...")
            ani.save(out_file, writer='pillow')
        
        print(f"Success! Animation saved as: {out_file}")

if __name__ == "__main__":
    main()