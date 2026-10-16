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
