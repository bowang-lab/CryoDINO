"""
Multi-sample PCA overview figure.

Layout: one row per sample; for each plane (Axial | Coronal | Sagittal) the
columns are  Orig | Label | PCA  (Label column is blank for samples without a label).
Each plane uses the slice with the most label foreground (middle slice if no label).

Usage:
    python visualization/plot_pca_grid.py \
        --vis-base ~/projects/cryoet_pca/vis_output_336/highres128 \
        --samples TS_0001 TS_0009 TS_027 \
        --label TS_0001=/path/labelsTr/TS_0001.nii.gz --label TS_0009=/path/... \
        --output ~/projects/cryoet_pca/figures_pca_grid/highres128_pca_grid.png \
        --contrast-thr 0.2 --bg-thr 0.3 --bg-thr-for TS_0009=0.25
"""

import argparse
import gc
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_all_heads import (best_plane_idx, get_pca_slice, get_slice,
                            process_label, vis_pca)

PLANES = ['Axial', 'Coronal', 'Sagittal']
AX_DIM = {'Axial': 2, 'Coronal': 1, 'Sagittal': 0}


def parse_kv(items, cast=str):
    out = {}
    for item in items or []:
        if '=' not in item:
            raise argparse.ArgumentTypeError(f'expected NAME=VALUE, got {item!r}')
        k, v = item.split('=', 1)
        out[k] = cast(v)
    return out


def sample_panels(name, args, labels, bg_thrs):
    """Load one sample and return its per-plane 2D slices (volumes are freed)."""
    pca_dir = os.path.join(args.vis_base, name, 'pca')
    orig_vol, pca_vol = vis_pca(
        pca_dir,
        crop=args.crop,
        bg_thr=bg_thrs.get(name, args.bg_thr),
        contrast_thr=args.contrast_thr,
        comps=args.pca_comps,
        bg_invert=args.bg_invert,
        no_bg_mask=args.no_bg_mask,
    )
    lbl_path = labels.get(name)
    lbl_vol, cmap, norm = (process_label(lbl_path, orig_vol.shape, args.crop)
                           if lbl_path else (None, None, None))

    lo, hi = np.percentile(orig_vol, 1), np.percentile(orig_vol, 99)
    img_disp = np.clip((orig_vol - lo) / (hi - lo + 1e-8), 0, 1)

    panels = {}
    for plane in PLANES:
        idx = (best_plane_idx(lbl_vol, plane) if lbl_vol is not None
               else orig_vol.shape[AX_DIM[plane]] // 2)
        panels[plane] = {
            'idx': idx,
            'img': get_slice(img_disp, plane, idx).copy(),
            'lbl': get_slice(lbl_vol, plane, idx).copy() if lbl_vol is not None else None,
            'pca': get_pca_slice(pca_vol, plane, idx).copy(),
        }
    del orig_vol, pca_vol, lbl_vol, img_disp
    gc.collect()
    return panels, cmap, norm


def main(args):
    labels = parse_kv(args.label)
    bg_thrs = parse_kv(args.bg_thr_for, float)
    samples = [s for s in args.samples if s not in set(args.exclude)]

    rows = []
    for name in samples:
        if not os.path.isdir(os.path.join(args.vis_base, name, 'pca')):
            print(f'SKIP {name}: no pca dir under {args.vis_base}')
            continue
        if name in labels and not os.path.isfile(labels[name]):
            print(f'WARNING {name}: label not found ({labels[name]}); plotting without label')
            labels.pop(name)
        print(f'Loading {name} (bg_thr={bg_thrs.get(name, args.bg_thr)}, label={labels.get(name)})')
        rows.append((name, *sample_panels(name, args, labels, bg_thrs)))
    if not rows:
        raise SystemExit('No samples to plot.')

    sub_cols = ['Orig', 'Label', 'PCA']
    n_rows, n_cols = len(rows), len(PLANES) * len(sub_cols)
    cell = args.cell_size
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False,
        figsize=(n_cols * cell + 0.6, n_rows * cell + 0.5),
        gridspec_kw={'hspace': 0.04, 'wspace': 0.02},
    )

    for r, (name, panels, cmap, norm) in enumerate(rows):
        for p, plane in enumerate(PLANES):
            pan = panels[plane]
            for s, kind in enumerate(sub_cols):
                ax = axes[r, p * len(sub_cols) + s]
                if kind == 'Orig':
                    ax.imshow(pan['img'], cmap='gray', vmin=0, vmax=1)
                elif kind == 'Label':
                    ax.imshow(pan['img'], cmap='gray', vmin=0, vmax=1)
                    if pan['lbl'] is not None:
                        ax.imshow(pan['lbl'], cmap=cmap, norm=norm, alpha=0.75,
                                  interpolation='nearest')
                else:
                    ax.imshow(pan['pca'], interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    title = f'{plane}\n{kind}' if s == 1 else kind
                    ax.set_title(title, fontsize=6, fontweight='bold', pad=3)
        axes[r, 0].set_ylabel(name, fontsize=6, rotation=0, labelpad=22,
                              va='center', color='#444')

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved -> {args.output}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--vis-base', required=True, help='Dir containing <sample>/pca/ (e.g. vis_output_336/highres128)')
    ap.add_argument('--samples', nargs='+', required=True)
    ap.add_argument('--exclude', nargs='*', default=[], help='Samples to drop from --samples')
    ap.add_argument('--label', action='append', help='NAME=label_path (repeatable); samples without one get a blank label cell')
    ap.add_argument('--output', required=True, help='Output PNG path')
    ap.add_argument('--crop', type=int, default=None)
    ap.add_argument('--bg-thr', type=float, default=0.5)
    ap.add_argument('--bg-thr-for', action='append', help='Per-sample override NAME=thr (repeatable)')
    ap.add_argument('--contrast-thr', type=float, default=0.4)
    ap.add_argument('--pca-comps', type=int, nargs=3, default=(1, 2, 3))
    ap.add_argument('--bg-invert', action='store_true')
    ap.add_argument('--no-bg-mask', action='store_true')
    ap.add_argument('--cell-size', type=float, default=1.0, help='Inches per panel')
    main(ap.parse_args())
