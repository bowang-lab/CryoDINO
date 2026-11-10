"""
Plot all MHSA attention heads + PCA RGB across 4 slices.

Layout: 4 rows × (2 or 3 fixed cols + n_heads) cols
  Rows : z at ~10%, ~50%, ~80% of depth  +  best foreground slice
  Cols : Original | [Label overlay] | PCA RGB | MHSA head 0 … head N-1

Usage:
    python visualization/plot_all_heads.py \
        --pca-dir  visualization/attention_maps/vis_output/TS_0001/pca_448 \
        --mhsa-dir visualization/attention_maps/vis_output/TS_0001/mhsa_448 \
        --output-dir visualization/attention_maps \
        --name TS_0001 \
        --lbl-path dataset/downstream/Dataset001_CZII_10001/labelsTr/TS_0001.nii.gz
"""

import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
from torch.nn.functional import interpolate
import os
from monai.transforms import Resize
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'DejaVu Serif'],
    'font.size':   7,
})

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _load_ds(path, size, mode):
    v = torch.from_numpy(np.asarray(nib.load(path).dataobj, dtype=np.float32))
    if tuple(v.shape) != tuple(size):
        v = interpolate(v[None, None], size=size, mode=mode)[0, 0]
    return v


def _crop_slices(shape, crop):
    if crop is None:
        return tuple(slice(None) for _ in shape)
    return tuple(
        slice(max(0, (s - crop) // 2), max(0, (s - crop) // 2) + min(crop, s))
        for s in shape
    )


def _load_crop(path, crop):
    img = nib.load(path)
    sl = _crop_slices(img.shape, crop)
    return torch.from_numpy(np.asarray(img.dataobj[sl], dtype=np.float32))


def vis_pca(pth, crop=None, bg_thr=0.5, contrast_thr=0.4, comps=(1, 2, 3), bg_invert=False, no_bg_mask=False):
    img = _load_crop(f'{pth}/orig.nii.gz', crop)
    pca0 = _load_crop(f'{pth}/nifti_0.nii.gz', crop)
    bg_mask = pca0 < bg_thr if bg_invert else pca0 > bg_thr
    del pca0
    pca_vol = torch.stack([_load_crop(f'{pth}/nifti_{i}.nii.gz', crop) for i in comps], dim=3)
    pca_min = pca_vol.reshape(-1, 3).min(dim=0).values.reshape(1, 1, 1, -1)
    pca_max = pca_vol.reshape(-1, 3).max(dim=0).values.reshape(1, 1, 1, -1)
    pca_vol = (pca_vol - pca_min) / (pca_max - pca_min)
    del pca_min, pca_max
    if not no_bg_mask:
        pca_vol[bg_mask, :] = 1.0
    del bg_mask
    if contrast_thr > 0:
        pca_vol[pca_vol < contrast_thr] = 0
    return img.numpy(), pca_vol.numpy()


def load_mhsa_heads(mhsa_dir, crop=None, only=None):
    """Return {head_idx: normalized volume}. `only` may be None (all), an int, or a set of ints."""
    if isinstance(only, int):
        only = {only}
    heads, i = {}, 0
    while os.path.exists(os.path.join(mhsa_dir, f'nifti_{i}.nii.gz')):
        if only is None or i in only:
            h = _load_crop(os.path.join(mhsa_dir, f'nifti_{i}.nii.gz'), crop).numpy()
            h = (h - h.min()) / (h.max() - h.min() + 1e-8)
            heads[i] = h
        i += 1
    if only is not None:
        missing = sorted(set(only) - set(heads))
        if missing:
            raise ValueError(f'Head(s) {missing} not found in {mhsa_dir} (found 0..{i - 1})')
    return heads

# def get_slice_indices(depth, lbl_vol=None):
#     """Return dict: begin / middle / end / best-10 / best / best+10."""
#     indices = {
#         'begin':  int(depth * 0.10),
#         'middle': int(depth * 0.50),
#         'end':    int(depth * 0.80),
#     }
#     if lbl_vol is not None:
#         fg_counts = (lbl_vol > 0).sum(axis=(0, 1))
#         best = int(np.argmax(fg_counts))
#     else:
#         best = int(depth * 0.65)
#     indices['best-10'] = max(0, best - 10)
#     indices['best']    = best
#     indices['best+10'] = min(depth - 1, best + 10)
#     return indices

def best_plane_idx(lbl_vol, plane):
    """Slice index with maximum foreground for a given plane."""
    ax  = {'Axial': 2, 'Coronal': 1, 'Sagittal': 0}[plane]
    cnt = (lbl_vol > 0).sum(axis=tuple(i for i in range(3) if i != ax))
    return int(np.argmax(cnt))


def get_slice(vol, plane, idx):
    if plane == 'Axial':
        s = vol[:, :, idx]
    elif plane == 'Coronal':
        s = vol[:, idx, :]
    else:
        s = vol[idx, :, :]
    return np.rot90(s)


def get_pca_slice(pca_vol, plane, idx):
    if plane == 'Axial':
        s = pca_vol[:, :, idx, :]
    elif plane == 'Coronal':
        s = pca_vol[:, idx, :, :]
    else:
        s = pca_vol[idx, :, :, :]
    return np.rot90(s)


# process lbl
def process_label(lbl_path, target_shape, crop=None):
    lbl_vol = _load_crop(lbl_path, crop).numpy()
    if lbl_vol.shape != tuple(target_shape):
        lbl_vol = Resize(tuple(target_shape), mode='nearest')(torch.tensor(lbl_vol).unsqueeze(0))[0].numpy()

    _lc = np.array([
    [0,    0,   0,   0.00],
    [0,  114, 178,   0.85],
    [213,  94,   0,  0.85],
    [0,  158, 115,   0.85],
    ], dtype=float)
    _lc[:, :3] /= 255.0
    LABEL_CMAP = ListedColormap(_lc)
    LABEL_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)

    return lbl_vol, LABEL_CMAP, LABEL_NORM

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main(args):
    # Load
    # orig_vol, pca_vol = vis_pca(args.pca_dir, resize=1.0)
    # heads = load_mhsa_heads(args.mhsa_dir, orig_vol.shape)
    orig_vol, pca_vol = vis_pca(
        args.pca_dir,
        crop=args.crop,
        bg_thr=args.bg_thr,
        contrast_thr=args.contrast_thr,
        comps=args.pca_comps,
        bg_invert=args.bg_invert,
        no_bg_mask=args.no_bg_mask,
    )
    lbl_vol, LABEL_CMAP, LABEL_NORM = process_label(args.lbl_path, orig_vol.shape, args.crop) if args.lbl_path else (None, None, None)
    PLANES = ['Axial', 'Coronal', 'Sagittal']

    # Which heads go in each row:
    #   --row-heads 8 12 0 -> row A uses H8, row B H12, row C H0 (one head column)
    #   --head-idx 10      -> every row uses H10
    #   neither            -> every row shows all heads
    if args.row_heads is not None:
        if len(args.row_heads) != len(PLANES):
            raise ValueError(f'--row-heads needs {len(PLANES)} values (one per row), got {len(args.row_heads)}')
        heads = load_mhsa_heads(args.mhsa_dir, crop=args.crop, only=set(args.row_heads))
        row_head_ids = [[h] for h in args.row_heads]
    else:
        heads = load_mhsa_heads(args.mhsa_dir, crop=args.crop, only=args.head_idx)
        row_head_ids = [sorted(heads)] * len(PLANES)
    single_head_col = args.row_heads is not None or args.head_idx is not None

    n_heads = len(row_head_ids[0])
    print(f'Loaded MHSA heads {sorted(heads)}  |  per-row: {row_head_ids}  |  volume shape: {orig_vol.shape}')

    # --- previous: z-only rows (begin/middle/end/best±10) ---
    # depth = orig_vol.shape[2]
    # s_idx  = get_slice_indices(depth, lbl_vol)
    # row_keys  = ['begin', 'middle', 'end', 'best-10', 'best', 'best+10']
    # z_values  = [s_idx[k] for k in row_keys]
    # n_rows  = 6

    # --- new: 3 rows — best slice per plane (Axial / Coronal / Sagittal) ---

    # Layout
    img_disp = (orig_vol - np.percentile(orig_vol, 1)) / (np.percentile(orig_vol, 99) - np.percentile(orig_vol, 1) + 1e-8)
    img_disp = np.clip(img_disp, 0, 1)

    has_lbl = lbl_vol is not None
    n_fixed = 3 if has_lbl else 2          # orig + [label] + pca
    n_cols  = n_fixed + n_heads
    n_rows  = len(PLANES)

    col_w, row_h = 1.4, 1.4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * col_w + 0.6, n_rows * row_h + 0.8),
        gridspec_kw={'hspace': 0.05, 'wspace': 0.02},
    )

    # Column headers
    headers = ['Orig'] + (['Label'] if has_lbl else []) + ['PCA'] + (['Attention Head'] if single_head_col else [f'H{i}' for i in row_head_ids[0]])
    for c, t in enumerate(headers):
        axes[0, c].set_title(t, fontsize=6, fontweight='bold', pad=3)

    # Fill rows — one per plane, best foreground slice
    for r, plane in enumerate(PLANES):
        if has_lbl:
            idx = best_plane_idx(lbl_vol, plane)
        else:
            ax_dim = {'Axial': 2, 'Coronal': 1, 'Sagittal': 0}[plane]
            idx = orig_vol.shape[ax_dim] // 2

        img_sl  = get_slice(img_disp, plane, idx)
        pca_sl  = get_pca_slice(pca_vol, plane, idx)
        row_lbl = f'{plane[0]}\n(={idx})'
        col = 0

        # --- Col 0: Original ---
        ax = axes[r, col]
        ax.imshow(img_sl, cmap='gray')
        ax.set_ylabel(row_lbl, fontsize=6, rotation=0, labelpad=32, va='center', color='#444')
        ax.set_yticks([]); ax.set_xticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        col += 1

        # --- Col 1 (optional): Label overlay ---
        if has_lbl:
            lbl_sl = get_slice(lbl_vol,  plane, idx)
            dsp_sl = get_slice(img_disp, plane, idx)
            ax = axes[r, col]
            ax.imshow(dsp_sl, cmap='gray', vmin=0, vmax=1)
            ax.imshow(lbl_sl, cmap=LABEL_CMAP, norm=LABEL_NORM, alpha=0.75, interpolation='nearest')
            ax.axis('off')
            col += 1

        # --- PCA RGB ---
        axes[r, col].imshow(pca_sl, interpolation='nearest')
        axes[r, col].axis('off')
        col += 1

        # --- MHSA heads ---
        for h_id in row_head_ids[r]:
            h_sl = get_slice(heads[h_id], plane, idx)
            axes[r, col].imshow(img_sl, cmap='gray')
            axes[r, col].imshow(h_sl, cmap='jet', alpha=0.5)
            axes[r, col].axis('off')
            if args.row_heads is not None:
                # tag which head this row uses, e.g. "H8", in the top-left corner
                axes[r, col].text(0.03, 0.97, f'H{h_id}', transform=axes[r, col].transAxes,
                                  fontsize=6, fontweight='bold', color='white', va='top', ha='left',
                                  bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))
            col += 1

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'{args.name}_best_planes.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pca-dir',    required=True,       help='Dir with PCA niftis + orig.nii.gz')
    parser.add_argument('--mhsa-dir',   required=True,       help='Dir with MHSA niftis')
    parser.add_argument('--output-dir', required=True,       help='Output directory')
    parser.add_argument('--name',       default='sample',    help='Sample name for output filename')
    parser.add_argument('--lbl-path',   default=None,        help='Optional label nifti path')
    parser.add_argument('--crop', type=int, default=None, help='Center-crop each axis to this size (voxels)')
    parser.add_argument('--head-idx', type=int, default=None, help='Plot only this attention head; default plots all heads')
    parser.add_argument('--row-heads', type=int, nargs=3, default=None,
                        help='One head per row (Axial Coronal Sagittal), e.g. --row-heads 8 12 0. Overrides --head-idx')
    parser.add_argument('--bg-thr', type=float, default=0.5, help='Threshold for the background mask')
    parser.add_argument('--contrast-thr', type=float, default=0.4, help='Contrast threshold for the PCA volume; <= 0 disables thresholding')
    parser.add_argument('--pca-comps', type=int, nargs=3, default=(1, 2, 3), help='PCA component indices used as RGB channels')
    parser.add_argument('--bg-invert', action='store_true', help='Invert the background mask threshold comparison')
    parser.add_argument('--no-bg-mask', action='store_true', help='Do not set the PCA background to white')
    main(parser.parse_args())
