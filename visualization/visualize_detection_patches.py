"""
Visualize detection patches with overlaid point annotations.

Loads .pt patches and their GT points from a czi_100_datalist.json, then
renders 3 orthogonal mid-plane slices (XY / XZ / YZ) for each patch.
Dots are colored per particle class and shown only within ±slice_tol voxels
of each mid-plane.

Usage:
    python visualization/visualize_detection_patches.py \
        --datalist  /path/to/czi_100_datalist.json \
        --output-dir /path/to/output_vis \
        [--n-patches 20] \
        [--split training|validation|test] \
        [--slice-tol 5] \
        [--seed 42]
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch


# ── 6-class CZI colour palette (alphabetical order matches auto class map) ──
CLASS_NAMES = [
    "Beta-amylase",        # 0
    "Beta-galactosidase",  # 1
    "Thyroglobulin",       # 2
    "cytosolic ribosome",  # 3
    "ferritin complex",    # 4
    "virus-like capsid",   # 5
]
COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#ff7f00",  # orange
    "#984ea3",  # purple
    "#ffff33",  # yellow
]


def load_patch(path: str) -> np.ndarray:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data.numpy().astype(np.float32)


def filter_real_points(points) -> np.ndarray:
    """Remove the padding sentinel row [-100,-100,-100,-100,-100]."""
    pts = np.array(points, dtype=np.float32)
    if len(pts) == 0:
        return pts
    valid = ~(pts[:, 0] < -50)
    return pts[valid]


def normalise_for_display(arr: np.ndarray) -> np.ndarray:
    p_lo, p_hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, p_lo, p_hi)
    rng = p_hi - p_lo
    if rng < 1e-8:
        return np.zeros_like(arr)
    return (arr - p_lo) / rng


def plot_patch(patch: np.ndarray, pts: np.ndarray, patch_name: str,
               out_path: str, slice_tol: int = 5):
    """
    3-panel figure: XY (mid-Z), XZ (mid-Y), YZ (mid-X) slices with dots.

    patch shape: (X, Y, Z) — nibabel XYZ convention.
    pts shape  : (N, 5) — [x, y, z, class_id, sigma], patch-local coords.
    """
    X, Y, Z = patch.shape
    mx, my, mz = X // 2, Y // 2, Z // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(patch_name, fontsize=9)

    panels = [
        # (slice_2d,        scatter_cols, scatter_rows, xlabel,   ylabel,   title)
        (normalise_for_display(patch[:, :, mz].T),   # XY plane at mid-Z
         pts, 0, 1, 2,  "X (vox)", "Y (vox)", f"XY  (Z={mz})"),
        (normalise_for_display(patch[:, my, :].T),   # XZ plane at mid-Y
         pts, 0, 2, 1,  "X (vox)", "Z (vox)", f"XZ  (Y={my})"),
        (normalise_for_display(patch[mx, :, :].T),   # YZ plane at mid-X
         pts, 1, 2, 0,  "Y (vox)", "Z (vox)", f"YZ  (X={mx})"),
    ]

    for ax, (img2d, all_pts, col_h, col_v, col_depth, xl, yl, title) in zip(axes, panels):
        ax.imshow(img2d, cmap="gray", origin="lower", aspect="equal")
        ax.set_xlabel(xl, fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)

        if len(all_pts) == 0:
            continue

        mid_depth = [X // 2, Y // 2, Z // 2][col_depth]
        for pt in all_pts:
            h_val    = pt[col_h]
            v_val    = pt[col_v]
            d_val    = pt[col_depth]
            cls      = int(round(pt[3]))
            sigma    = max(float(pt[4]), 2.0)

            # Only show dots near this slice
            if abs(d_val - mid_depth) > slice_tol:
                continue

            color  = COLORS[cls % len(COLORS)]
            radius = sigma  # draw circle sized to particle sigma
            circle = plt.Circle(
                (h_val, v_val), radius,
                color=color, fill=False, linewidth=1.0, alpha=0.85,
            )
            ax.add_patch(circle)
            ax.plot(h_val, v_val, "+", color=color, markersize=4, markeredgewidth=0.8)

    # Legend — only classes present in this patch
    present_cls = set(int(round(p[3])) for p in pts) if len(pts) > 0 else set()
    legend_handles = [
        mpatches.Patch(color=COLORS[c % len(COLORS)],
                       label=f"{c}: {CLASS_NAMES[c] if c < len(CLASS_NAMES) else c}")
        for c in sorted(present_cls)
    ]
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(len(legend_handles), 3), fontsize=7,
                   bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist",   required=True, help="Path to czi_100_datalist.json")
    parser.add_argument("--output-dir", required=True, help="Directory to save visualizations")
    parser.add_argument("--n-patches",  type=int, default=20,
                        help="Number of patches to visualize (default: 20). -1 for all.")
    parser.add_argument("--split",      default="training",
                        choices=["training", "validation", "test"],
                        help="Which split to sample from (default: training)")
    parser.add_argument("--slice-tol",  type=int, default=5,
                        help="±voxel tolerance around mid-plane to show dots (default: 5)")
    parser.add_argument("--only-particles", action="store_true",
                        help="Only visualize patches that contain at least one particle")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.datalist) as f:
        datalist = json.load(f)

    entries = datalist[args.split]
    print(f"Split '{args.split}': {len(entries)} entries total")

    if args.only_particles:
        entries = [e for e in entries if filter_real_points(e["points"]).shape[0] > 0]
        print(f"  → {len(entries)} entries with at least 1 particle")

    if args.n_patches > 0 and args.n_patches < len(entries):
        entries = random.sample(entries, args.n_patches)

    print(f"Visualizing {len(entries)} patches → {args.output_dir}")

    for i, entry in enumerate(entries):
        img_path = entry["image"]
        pts      = filter_real_points(entry["points"])

        if not os.path.exists(img_path):
            print(f"  [{i+1}/{len(entries)}] SKIP (missing): {img_path}")
            continue

        patch_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path   = os.path.join(args.output_dir, f"{patch_name}.png")

        patch = load_patch(img_path)
        plot_patch(patch, pts, patch_name, out_path, slice_tol=args.slice_tol)

        n_pts = len(pts)
        print(f"  [{i+1}/{len(entries)}] {patch_name}  ({n_pts} particles)  → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
