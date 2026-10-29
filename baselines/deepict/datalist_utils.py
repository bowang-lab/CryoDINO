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


def remap_12049(label_arr):
    """Dataset049/EMPIAR-12049 raw {0..5} -> remapped {0,1,2,3}.

    Matches `_remap_12049` in `3DINO/dinov2/eval/segmentation_3d/augmentations.py` exactly: raw
    classes 1 (matrix) and 5 (centrioles) merge into background, the rest renumber contiguously
    (2->1 membranes, 3->2 microtubules, 4->3 ribosomes).
    """
    out = np.zeros_like(label_arr)
    out[label_arr == 2] = 1
    out[label_arr == 3] = 2
    out[label_arr == 4] = 3
    return out.astype(label_arr.dtype)


def foreground_values(label_arr):
    """Sorted non-zero label values present in one volume, as ints."""
    return [int(v) for v in np.unique(label_arr) if int(v) != 0]


def split_classes(label_arr, class_values):
    """Multi-label volume -> one binary uint8 mask per class value, in the given order.

    DeePiCt trains one output channel per semantic class and reads a *separate* mask volume for
    each (see generate_training_data.py), so a multi-class label has to be split into N files
    rather than passed through as a single multi-valued volume.

    A class absent from this particular tomogram still yields an all-zero mask: the
    <class>_mask column must resolve to a real file for every row of metadata.csv.
    """
    return [(label_arr == v).astype(np.uint8) for v in class_values]


def save_mrc(arr, path, dtype=np.float32):
    """Write a numpy array to .mrc (membrain's segment reads MRC only, not NIfTI)."""
    import mrcfile
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(arr, dtype=dtype))

