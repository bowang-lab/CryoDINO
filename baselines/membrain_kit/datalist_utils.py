"""Shared helpers for the JSON-datalist driver (paths used verbatim from the JSON)."""
import json
import os

import numpy as np


def load_datalist(path):
    """Return dict with 'training'/'validation'/'test' lists of {'image','label'} entries."""
    with open(path) as f:
        d = json.load(f)
    for k in ("training", "validation", "test"):
        d.setdefault(k, [])
    return d


def strip_ext(fname):
    """'UF5.nii.gz' -> 'UF5' ; 'TE7_0000_patch_0_0_0.pt' -> 'TE7_0000_patch_0_0_0'."""
    b = os.path.basename(fname)
    for ext in (".nii.gz", ".nii", ".mrc", ".rec", ".pt", ".h5"):
        if b.endswith(ext):
            return b[: -len(ext)]
    return os.path.splitext(b)[0]


def load_volume(path):
    """Load a .pt (torch tensor) or .nii.gz/.mrc volume as a float32 numpy array."""
    if path.endswith(".pt"):
        import torch
        return torch.load(path, map_location="cpu").numpy().astype(np.float32)
    if path.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        return np.asarray(nib.load(path).dataobj).astype(np.float32)
    if path.endswith((".mrc", ".rec")):
        import mrcfile
        with mrcfile.open(path, permissive=True) as m:
            return np.asarray(m.data).astype(np.float32)
    raise ValueError(f"unsupported volume format: {path}")


def binarize(label_arr):
    """CZII multi-class {0,1,2,3} -> binary foreground {0,1} (uint8)."""
    return (label_arr > 0).astype(np.uint8)


def save_mrc(arr, path, dtype=np.float32):
    """Write a numpy array to .mrc (membrain's segment reads MRC only, not NIfTI)."""
    import mrcfile
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(arr, dtype=dtype))

