"""Shared helpers for the JSON-datalist driver (detection flavour: points, not masks).

Dataset440-style datalist entries look like

    {"image": ".../images/TS_5_4_0_64_56.pt",
     "points": [[p0, p1, p2, class, radius], ...],
     "voxel_size": 10.012}

where the .pt is a 128^3 cube cropped out of a whole tomogram at the offset encoded in its
name, and a point's (p0, p1, p2) indexes that cube directly (v[p0, p1, p2] is the particle).
Rows whose class is negative are the "no annotation" sentinel (-100) and carry no particle.
"""
import json
import os
import re

import numpy as np

# TS_5_4_0_64_56  ->  ('TS_5_4', 0, 64, 56)
PATCH_RE = re.compile(r"^(TS_\d+_\d+)_(\d+)_(\d+)_(\d+)$")


def load_datalist(path):
    """Return dict with 'training'/'validation'/'test' lists (missing splits become [])."""
    with open(path) as f:
        d = json.load(f)
    for k in ("training", "validation", "test"):
        d.setdefault(k, [])
    return d


def strip_ext(fname):
    """'TS_5_4_0_64_56.pt' -> 'TS_5_4_0_64_56' ; 'TS_100_3_0000.nii.gz' -> 'TS_100_3_0000'."""
    b = os.path.basename(fname)
    for ext in (".nii.gz", ".nii", ".mrc", ".rec", ".pt", ".h5"):
        if b.endswith(ext):
            return b[: -len(ext)]
    return os.path.splitext(b)[0]


def parse_patch_name(name):
    """'TS_5_4_0_64_56' -> ('TS_5_4', (0, 64, 56)). Raises ValueError on anything else."""
    m = PATCH_RE.match(name)
    if m is None:
        raise ValueError(f"not a Dataset440 patch name: {name!r}")
    return m.group(1), (int(m.group(2)), int(m.group(3)), int(m.group(4)))


def load_patch(path, image_root=None):
    """Load a .pt cube as float32. The datalist records the paths of the machine it was built
    on, so resolve by basename under image_root when one is given."""
    import torch

    if image_root:
        path = os.path.join(image_root, os.path.basename(path))
    return torch.load(path, map_location="cpu").numpy().astype(np.float32)


def resolve_path(path, root=None):
    """Datalist paths are used verbatim (as in the deepict/membrain kits). If the file is not there
    and a root is given, fall back to <root>/<basename> -- for datalists built on another machine."""
    if os.path.exists(path) or not root:
        return path
    return os.path.join(root, os.path.basename(path))


def load_nifti_xyz(path):
    """Load a whole-tomogram .nii/.nii.gz as float32, in nibabel's native (X, Y, Z) order.

    Same read as CryoDINO's detection loader (detection_3d/augmentations.py
    `_load_detection_image`: nib.load(x).get_fdata()), so a datalist point (p0, p1, p2) indexes
    this array as vol[p0, p1, p2] -- exactly the convention the .pt patches use.
    """
    import nibabel as nib

    return nib.load(path).get_fdata(dtype=np.float32)


def standardize(vol):
    """DeepETPicker's own 'standardization' (utils/normalization.py): float32, x -= mean, x /= std
    over the whole tomogram. Used when writing data_std/ directly for inference, so inference
    inputs are normalised exactly as preprocess.py normalises the training tomograms."""
    vol = np.array(vol, dtype=np.float32, copy=True)
    vol -= vol.mean()
    vol /= vol.std()
    return vol


def valid_points(points):
    """Datalist points -> (N, 5) float array with the -100 'no annotation' rows removed."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 5)
    return p[p[:, 3] >= 0]


def check_points_in_volume(points, shape_xyz, name):
    """Fail loudly if any labelled point lies outside the volume: the cheap test for an axis-order
    mismatch between the NIfTI and the datalist points."""
    p = valid_points(points)
    if len(p) == 0:
        return 0
    lo = (p[:, :3] < -0.5).any(axis=1)
    hi = (p[:, :3] > np.asarray(shape_xyz) - 0.5).any(axis=1)
    bad = int((lo | hi).sum())
    if bad:
        raise SystemExit(f"{name}: {bad}/{len(p)} points fall outside volume shape {tuple(shape_xyz)} "
                         f"(point max {p[:, :3].max(axis=0)}) -- axis order of NIfTI vs points differs")
    return len(p)


def save_mrc(arr, path, dtype=np.float32):
    """Write a numpy array to .mrc (DeepETPicker reads MRC only)."""
    import mrcfile

    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(arr, dtype=dtype))


def read_coords(path):
    """Read a DeepETPicker .coords file -> (N, >=4) float array of (cls, x, y, z, ...)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 5), dtype=np.float64)
    a = np.loadtxt(path, delimiter="\t", ndmin=2)
    return a.astype(np.float64)
