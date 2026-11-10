"""
Multi-sample PCA + attention-head overview figure.

Layout: one row per (sample, plane) cut, e.g. "TS_0001:A".
Columns: Orig | Label | PCA | H0 ... H15   (or only the heads given with --heads)

Planes: A = Axial, C = Coronal, S = Sagittal. Each cut uses the slice with the
most label foreground in that plane (middle slice if the sample has no label).

Usage:
    python visualization/plot_pca_grid.py \
        --vis-base ~/projects/cryoet_pca/vis_output_336/highres128 \
        --rows TS_0001:A TS_0001:C TS_0008:S TS_034:A \
        --label TS_0001=/path/labelsTr/TS_0001.nii.gz --label TS_0008=/path/... \
        --output ~/projects/cryoet_pca/figures_pca_grid/highres128_pca_grid.png \
        --contrast-thr 0.2 --bg-thr 0.3 --bg-thr-for TS_0008=0.25
"""

import argparse
import gc
import os
from collections import OrderedDict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_all_heads import (_load_crop, best_plane_idx, get_pca_slice,
                            get_slice, process_label, vis_pca)

PLANE_CODES = {'A': 'Axial', 'C': 'Coronal', 'S': 'Sagittal'}
AX_DIM = {'Axial': 2, 'Coronal': 1, 'Sagittal': 0}


def parse_kv(items, cast=str):
    out = {}
    for item in items or []:
        if '=' not in item:
            raise argparse.ArgumentTypeError(f'expected NAME=VALUE, got {item!r}')
        k, v = item.split('=', 1)
        out[k] = cast(v)
    return out


def parse_rows(items):
    """'TS_0001:A' -> ('TS_0001', 'Axial'); a bare name expands to A, C, S."""
    rows = []
    for item in items:
        name, _, code = item.rpartition(':') if ':' in item else (item, '', '')
        if not code:
            rows += [(item, p) for p in PLANE_CODES.values()]
            continue
        if code.upper() not in PLANE_CODES:
            raise SystemExit(f'Bad plane in {item!r}; use A, C or S')
        rows.append((name, PLANE_CODES[code.upper()]))
    return rows


def list_heads(mhsa_dir):
    i = 0
    while os.path.exists(os.path.join(mhsa_dir, f'nifti_{i}.nii.gz')):
        i += 1
    return list(range(i))


def sample_cuts(name, planes, head_ids, args, labels, bg_thrs):
    """Load one sample once and return {plane: 2D panels} for the requested planes."""
    sample_dir = os.path.join(args.vis_base, name)
    orig_vol, pca_vol = vis_pca(
        os.path.join(sample_dir, 'pca'),
        crop=args.crop,
        bg_thr=bg_thrs.get(name, args.bg_thr),
        contrast_thr=args.contrast_thr,
        comps=args.pca_comps,
        bg_invert=args.bg_invert or name in args.bg_invert_for,
        no_bg_mask=args.no_bg_mask,
    )
    lbl_path = labels.get(name)
    lbl_vol, cmap, norm = (process_label(lbl_path, orig_vol.shape, args.crop)
                           if lbl_path else (None, None, None))

    lo, hi = np.percentile(orig_vol, 1), np.percentile(orig_vol, 99)
    img_disp = np.clip((orig_vol - lo) / (hi - lo + 1e-8), 0, 1)

    cuts = {}
    for plane in planes:
        idx = (best_plane_idx(lbl_vol, plane) if lbl_vol is not None
               else orig_vol.shape[AX_DIM[plane]] // 2)
        cuts[plane] = {
            'idx': idx,
            'img': get_slice(img_disp, plane, idx).copy(),
            'lbl': get_slice(lbl_vol, plane, idx).copy() if lbl_vol is not None else None,
            'pca': get_pca_slice(pca_vol, plane, idx).copy(),
            'heads': {},
            'cmap': cmap, 'norm': norm,
        }
    del orig_vol, pca_vol, lbl_vol, img_disp
    gc.collect()

    # Heads: load one volume at a time, normalize over the whole volume, keep only slices
    mhsa_dir = os.path.join(sample_dir, 'mhsa')
    for h in head_ids:
        path = os.path.join(mhsa_dir, f'nifti_{h}.nii.gz')
        if not os.path.exists(path):
            print(f'WARNING {name}: head {h} missing ({path})')
            continue
        hv = _load_crop(path, args.crop).numpy()
        hv = (hv - hv.min()) / (hv.max() - hv.min() + 1e-8)
        for plane in planes:
            cuts[plane]['heads'][h] = get_slice(hv, plane, cuts[plane]['idx']).copy()
        del hv
        gc.collect()
    return cuts


def blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def main(args):
    labels = parse_kv(args.label)
    bg_thrs = parse_kv(args.bg_thr_for, float)
    excluded = set(args.exclude)
    rows = [(n, p) for n, p in parse_rows(args.rows) if n not in excluded]

    # Group requested planes by sample so each sample is loaded only once
    by_sample = OrderedDict()
    for name, plane in rows:
        by_sample.setdefault(name, [])
        if plane not in by_sample[name]:
            by_sample[name].append(plane)

    head_ids = None
    cuts = {}
    for name, planes in by_sample.items():
        if not os.path.isdir(os.path.join(args.vis_base, name, 'pca')):
            print(f'SKIP {name}: no pca dir under {args.vis_base}')
            continue
        if name in labels and not os.path.isfile(labels[name]):
            print(f'WARNING {name}: label not found ({labels[name]}); plotting without label')
            labels.pop(name)
        if head_ids is None:
            head_ids = args.heads if args.heads is not None else list_heads(
                os.path.join(args.vis_base, name, 'mhsa'))
        print(f'Loading {name} planes={planes} bg_thr={bg_thrs.get(name, args.bg_thr)} '
              f'bg_invert={args.bg_invert or name in args.bg_invert_for} '
              f'label={labels.get(name)}')
        for plane, panels in sample_cuts(name, planes, head_ids, args, labels, bg_thrs).items():
            cuts[(name, plane)] = panels

    rows = [r for r in rows if r in cuts]
    if not rows:
        raise SystemExit('No rows to plot.')

    headers = ['Orig', 'Label', 'PCA'] + [f'H{h}' for h in head_ids]
    n_rows, n_cols = len(rows), len(headers)
    cell = args.cell_size
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False,
        figsize=(n_cols * cell + 0.8, n_rows * cell + 0.4),
        gridspec_kw={'hspace': 0.04, 'wspace': 0.02},
    )
    for c, t in enumerate(headers):
        axes[0, c].set_title(t, fontsize=6, fontweight='bold', pad=3)

    for r, (name, plane) in enumerate(rows):
        pan = cuts[(name, plane)]
        # Orig
        ax = axes[r, 0]
        ax.imshow(pan['img'], cmap='gray', vmin=0, vmax=1)
        ax.set_ylabel(f'{name}\n{plane[0]} (={pan["idx"]})', fontsize=6, rotation=0,
                      labelpad=26, va='center', color='#444')
        blank(ax)
        # Label
        ax = axes[r, 1]
        ax.imshow(pan['img'], cmap='gray', vmin=0, vmax=1)
        if pan['lbl'] is not None:
            ax.imshow(pan['lbl'], cmap=pan['cmap'], norm=pan['norm'], alpha=0.75,
                      interpolation='nearest')
        blank(ax)
        # PCA
        ax = axes[r, 2]
        ax.imshow(pan['pca'], interpolation='nearest')
        blank(ax)
        # Heads
        for c, h in enumerate(head_ids, start=3):
            ax = axes[r, c]
            ax.imshow(pan['img'], cmap='gray')
            if h in pan['heads']:
                ax.imshow(pan['heads'][h], cmap='jet', alpha=0.5)
            blank(ax)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved -> {args.output}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--vis-base', required=True, help='Dir containing <sample>/pca and <sample>/mhsa')
    ap.add_argument('--rows', nargs='+', required=True,
                    help='Cuts in display order: NAME:A / NAME:C / NAME:S (bare NAME = all three)')
    ap.add_argument('--exclude', nargs='*', default=[], help='Samples to drop')
    ap.add_argument('--heads', type=int, nargs='+', default=None, help='Head columns to show (default: all)')
    ap.add_argument('--label', action='append', help='NAME=label_path (repeatable)')
    ap.add_argument('--output', required=True, help='Output PNG path')
    ap.add_argument('--crop', type=int, default=None)
    ap.add_argument('--bg-thr', type=float, default=0.5)
    ap.add_argument('--bg-thr-for', action='append', help='Per-sample override NAME=thr (repeatable)')
    ap.add_argument('--contrast-thr', type=float, default=0.4)
    ap.add_argument('--pca-comps', type=int, nargs=3, default=(1, 2, 3))
    ap.add_argument('--bg-invert', action='store_true', help='Invert the background mask for every sample')
    ap.add_argument('--bg-invert-for', nargs='*', default=[], help='Invert the background mask only for these samples')
    ap.add_argument('--no-bg-mask', action='store_true')
    ap.add_argument('--cell-size', type=float, default=0.8, help='Inches per panel')
    main(ap.parse_args())
