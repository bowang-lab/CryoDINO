"""
Visualize full tomograms from imagesTr with overlaid CSV point annotation blobs.

Run BEFORE patchify_detection_czi.sh to sanity-check raw tomograms and their
annotations prior to patch generation.

Per tomogram, saves two PNGs:
  {run}_ortho.png   — XY / XZ / YZ orthogonal mid-plane views with blob circles
  {run}_mosaic.png  — 4×4 mosaic of evenly-spaced Z slices with annotation dots

Usage:
    python visualization/detection_tomogram_overlay.py \
        --images-dir /path/to/Dataset440_CZII_10440/imagesTr \
        --csv        /path/to/point_annotations.csv \
        --output-dir /path/to/output_vis \
        --sigmas-ang '{"Beta-amylase":65,"Beta-galactosidase":90,"Thyroglobulin":130,"cytosolic ribosome":150,"ferritin complex":60,"virus-like capsid":135}' \
        [--n-tomos 5] \
        [--slice-tol 3] \
        [--seed 42]
"""

import argparse
import csv
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


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


# ---------------------------------------------------------------------------
# Helpers shared with downstream_patch_generation.py
# ---------------------------------------------------------------------------

def run_name_from_nii(nii_path: str) -> str:
    """'TS_5_4_0000.nii.gz' → 'TS_5_4'  (strips nnU-Net _XXXX channel suffix)."""
    base = os.path.basename(nii_path).replace(".nii.gz", "").replace(".nii", "")
    if "_" in base and base[-5] == "_" and base[-4:].isdigit():
        base = base[:-5]
    return base


def load_annotations(csv_path: str, sigmas_ang: dict, default_sigma_ang: float) -> tuple:
    """Load CSV point annotations; return (annotations, class_map).

    CSV format: run, particle_name, z, y, x, voxel_size  (z/y/x already in voxels).
    Stored as (x_vox, y_vox, z_vox, class_id, sigma_vox) — nibabel XYZ convention.
    """
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    particle_names = sorted({r["particle_name"].strip() for r in rows})
    class_map = {name: idx for idx, name in enumerate(particle_names)}
    print(f"  Auto class map: {class_map}")

    annotations = {}
    for row in rows:
        run      = row["run"].strip()
        particle = row["particle_name"].strip()
        z_vox    = float(row["z"])
        y_vox    = float(row["y"])
        x_vox    = float(row["x"])
        vs       = float(row["voxel_size"])

        class_id  = float(class_map[particle])
        sigma_vox = sigmas_ang.get(particle, default_sigma_ang) / vs

        if run not in annotations:
            annotations[run] = {"points": [], "voxel_size": vs}
        annotations[run]["points"].append([x_vox, y_vox, z_vox, class_id, sigma_vox])

    for run, info in annotations.items():
        print(f"  {run}: {len(info['points'])} particles  voxel_size={info['voxel_size']} Å/vox")

    return annotations, class_map


def normalise_for_display(arr: np.ndarray) -> np.ndarray:
    p_lo, p_hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, p_lo, p_hi)
    rng = p_hi - p_lo
    if rng < 1e-8:
        return np.zeros_like(arr)
    return (arr - p_lo) / rng


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_blobs_on_ax(ax, pts: np.ndarray, col_h: int, col_v: int, col_depth: int,
                      mid_depth: float, slice_tol: int):
    """Overlay blob circles on a 2D axes slice.

    pts      : (N, 5) array — [x, y, z, class_id, sigma_vox]
    col_h/v  : which column index maps to horizontal/vertical image axis
    col_depth: depth axis index (used for proximity filter)
    mid_depth: mid-plane depth value for proximity filter
    """
    if len(pts) == 0:
        return
    for pt in pts:
        if abs(pt[col_depth] - mid_depth) > slice_tol:
            continue
        h_val  = pt[col_h]
        v_val  = pt[col_v]
        cls    = int(round(pt[3]))
        sigma  = max(float(pt[4]), 2.0)
        color  = COLORS[cls % len(COLORS)]
        ax.add_patch(plt.Circle((h_val, v_val), sigma,
                                color=color, fill=False, linewidth=1.2, alpha=0.85))
        ax.plot(h_val, v_val, "+", color=color, markersize=5, markeredgewidth=0.9)


def _legend_handles(pts: np.ndarray) -> list:
    if len(pts) == 0:
        return []
    present = sorted({int(round(p[3])) for p in pts})
    return [
        mpatches.Patch(
            color=COLORS[c % len(COLORS)],
            label=f"{c}: {CLASS_NAMES[c] if c < len(CLASS_NAMES) else c}",
        )
        for c in present
    ]


def save_ortho(vol: np.ndarray, pts: np.ndarray, run: str,
               out_path: str, slice_tol: int):
    """3-panel orthogonal mid-plane figure: XY / XZ / YZ."""
    X, Y, Z = vol.shape
    mx, my, mz = X // 2, Y // 2, Z // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{run}  —  orthogonal mid-plane views", fontsize=9)

    panels = [
        # (img2d,                           col_h, col_v, col_depth, mid_depth, xlabel, ylabel, title)
        (normalise_for_display(vol[:, :, mz].T), 0, 1, 2, mz, "X (vox)", "Y (vox)", f"XY  (Z={mz})"),
        (normalise_for_display(vol[:, my, :].T), 0, 2, 1, my, "X (vox)", "Z (vox)", f"XZ  (Y={my})"),
        (normalise_for_display(vol[mx, :, :].T), 1, 2, 0, mx, "Y (vox)", "Z (vox)", f"YZ  (X={mx})"),
    ]

    for ax, (img2d, col_h, col_v, col_depth, mid_depth, xl, yl, title) in zip(axes, panels):
        ax.imshow(img2d, cmap="gray", origin="lower", aspect="equal")
        ax.set_xlabel(xl, fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
        _draw_blobs_on_ax(ax, pts, col_h, col_v, col_depth, mid_depth, slice_tol)

    handles = _legend_handles(pts)
    if handles:
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(handles), 3), fontsize=7, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_mosaic(vol: np.ndarray, pts: np.ndarray, run: str,
                out_path: str, slice_tol: int, n_cols: int = 4, n_rows: int = 4):
    """4×4 mosaic of evenly-spaced Z slices with annotation dots."""
    Z = vol.shape[2]
    n_panels = n_cols * n_rows
    z_indices = np.linspace(0, Z - 1, n_panels, dtype=int)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle(f"{run}  —  Z-slice mosaic", fontsize=10)

    for ax, z_idx in zip(axes.flat, z_indices):
        img2d = normalise_for_display(vol[:, :, z_idx].T)
        ax.imshow(img2d, cmap="gray", origin="lower", aspect="equal")
        ax.set_title(f"Z={z_idx}", fontsize=7)
        ax.tick_params(labelsize=5)
        _draw_blobs_on_ax(ax, pts, col_h=0, col_v=1, col_depth=2,
                          mid_depth=float(z_idx), slice_tol=slice_tol)

    handles = _legend_handles(pts)
    if handles:
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(handles), 3), fontsize=8, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Overlay point annotation blobs on raw tomograms (pre-patchify sanity check)."
    )
    parser.add_argument("--images-dir", required=True,
                        help="Directory containing tomogram NIfTI files (e.g. imagesTr/)")
    parser.add_argument("--csv", required=True,
                        help="Point annotations CSV (run, particle_name, z, y, x, voxel_size)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save visualizations")
    parser.add_argument("--sigmas-ang", default=None,
                        help='JSON mapping particle_name→radius_Å, e.g. \'{"ferritin complex":60}\'')
    parser.add_argument("--default-sigma-ang", type=float, default=60.0,
                        help="Fallback sigma in Å for particles not in --sigmas-ang (default: 60)")
    parser.add_argument("--slice-tol", type=int, default=3,
                        help="±voxel tolerance around the slice plane to show a blob (default: 3)")
    parser.add_argument("--n-tomos", type=int, default=-1,
                        help="Number of tomograms to visualize (-1 for all, default: -1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    sigmas_ang = json.loads(args.sigmas_ang) if args.sigmas_ang else {}

    # ── Collect NIfTI paths ────────────────────────────────────────────────
    nii_paths = sorted(
        os.path.join(args.images_dir, f)
        for f in os.listdir(args.images_dir)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    )
    if not nii_paths:
        raise FileNotFoundError(f"No NIfTI files found in {args.images_dir}")
    print(f"Found {len(nii_paths)} tomogram(s) in {args.images_dir}")

    if args.n_tomos > 0 and args.n_tomos < len(nii_paths):
        nii_paths = random.sample(nii_paths, args.n_tomos)
        print(f"Randomly selected {args.n_tomos} tomogram(s) (seed={args.seed})")

    # ── Load CSV annotations ───────────────────────────────────────────────
    print(f"\nLoading annotations from {args.csv} ...")
    annotations, class_map = load_annotations(args.csv, sigmas_ang, args.default_sigma_ang)

    # ── Per-tomogram visualization ─────────────────────────────────────────
    print(f"\nVisualizing {len(nii_paths)} tomogram(s) → {args.output_dir}")
    for i, nii_path in enumerate(nii_paths, 1):
        run = run_name_from_nii(nii_path)

        vol = nib.load(nii_path).get_fdata().astype(np.float32)
        X, Y, Z = vol.shape

        run_info = annotations.get(run, {"points": [], "voxel_size": None})
        pts = (np.array(run_info["points"], dtype=np.float32)
               if run_info["points"] else np.zeros((0, 5), dtype=np.float32))

        print(f"  [{i}/{len(nii_paths)}] {run}  shape={vol.shape}  "
              f"annotations={len(pts)}")

        ortho_path  = os.path.join(args.output_dir, f"{run}_ortho.png")
        mosaic_path = os.path.join(args.output_dir, f"{run}_mosaic.png")

        save_ortho(vol, pts, run, ortho_path, slice_tol=args.slice_tol)
        save_mosaic(vol, pts, run, mosaic_path, slice_tol=args.slice_tol)

        print(f"    → {ortho_path}")
        print(f"    → {mosaic_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
