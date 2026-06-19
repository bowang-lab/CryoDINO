# edited by Ahmadreza Attarpour; detection augmentations for CryoET 3D anchor-free detection
#
# Mirrors segmentation_3d/augmentations.py in structure, with "label" → "points".
#
# Flip/rotate logic ported from:
#   detection_repo_kaggle/Kaggle-2024-CryoET/cryoet/data/augmentations/functional.py
#   (random_flip_volume, rotate_and_scale_volume)
# and wrapped in MONAI MapTransform style.
#
# MONAI's RandFlipd / RandRotate90d treat any array as a spatial tensor and cannot
# correctly update raw Nx5 coordinate arrays (verified by test_augmentations.py).
# ApplyTransformToPointsd is also not suitable — it reads the NIfTI voxel-to-world
# affine, not the augmentation transform applied to the image.
#
# Data dict format:
#   "image"  : torch.Tensor [1, D, H, W]  (z-score normalised .pt patch, channel-first)
#   "points" : np.ndarray [N, 5]          columns = (x, y, z, class_id, sigma_vox)
#              padding rows use class_id = -100 as sentinel

import numpy as np
import torch

from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAdjustContrastd,
    OneOf,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandGibbsNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    ScaleIntensityRangePercentilesd,
    Lambdad,
    MapTransform,
    RandomizableTransform,
)


# ---------------------------------------------------------------------------
# Custom joint transforms: image + point coordinates
# Ported from Kaggle random_flip_volume / rotate_and_scale_volume
# ---------------------------------------------------------------------------

class RandFlipWithPointsd(RandomizableTransform, MapTransform):
    """Randomly flip image along one spatial axis and mirror point coordinates.

    Ported from Kaggle random_flip_volume — same logic, MONAI dict-transform wrapper.

    Points columns: (x, y, z, class_id, sigma_vox)   [axis → column]
       spatial_axis 0 → x → column 0
       spatial_axis 1 → y → column 1
       spatial_axis 2 → z → column 2
    Rows with class_id < 0 (column 3) are padding and passed through unchanged.

    Args:
        image_key    : dict key for image [1, D, H, W]
        points_key   : dict key for points [N, 5]
        spatial_axis : 0=z, 1=y, 2=x  (0-indexed spatial axis)
        prob         : flip probability
    """
    def __init__(self, image_key: str, points_key: str,
                 spatial_axis: int, prob: float = 0.5):
        MapTransform.__init__(self, keys=[image_key, points_key])
        RandomizableTransform.__init__(self, prob=prob)
        self.image_key = image_key
        self.points_key = points_key
        self.axis = spatial_axis

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        img = d[self.image_key]          # [1, D, H, W]
        pts = np.asarray(d[self.points_key]).copy()  # [N, 5] ndarray (handles MetaTensor/list)

        # Spatial size along this axis (dim 0 is channel → spatial dim = axis + 1)
        size = img.shape[self.axis + 1]

        # Flip image (same axis shift as above)
        d[self.image_key] = torch.flip(img, dims=[self.axis + 1])

        # Mirror coordinate: new_coord = (size - 1) - old_coord  [Kaggle convention]
        # AA: old sentinel (ZYX: column 0 = z = -100 for padding):
        # valid = pts[:, 0] >= 0
        valid = pts[:, 3] >= 0   # XYZ: padding rows have class_id = -100 at column 3
        pts[valid, self.axis] = (size - 1) - pts[valid, self.axis]
        d[self.points_key] = pts
        return d


class RandRotate90WithPointsd(RandomizableTransform, MapTransform):
    """Rotate image by k×90° (CCW) and update point coordinates.

    Coordinate transform for CCW rotation k times in spatial plane (a, b)
    — derived from torch.rot90 convention (verified by test_augmentations.py):
        k=1: new_a = size_b - 1 - old_b,  new_b = old_a
        k=2: new_a = size_a - 1 - old_a,  new_b = size_b - 1 - old_b
        k=3: new_a = old_b,               new_b = size_a - 1 - old_a

    Works for non-cubic patches (size_a may differ from size_b).

    Args:
        image_key    : dict key for image [1, D, H, W]
        points_key   : dict key for points [N, 5]
        spatial_axes : (a, b) — 0-indexed spatial axes for the rotation plane
        prob         : probability of applying a rotation (k uniform in {1,2,3})
    """
    def __init__(self, image_key: str, points_key: str,
                 spatial_axes: tuple = (0, 1), prob: float = 0.5):
        MapTransform.__init__(self, keys=[image_key, points_key])
        RandomizableTransform.__init__(self, prob=prob)
        self.image_key = image_key
        self.points_key = points_key
        self.a, self.b = spatial_axes
        self._k = 1

    def randomize(self, data):
        super().randomize(data)
        if self._do_transform:
            self._k = int(self.R.randint(1, 4))   # uniform in {1, 2, 3}

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        img = d[self.image_key]          # [1, D, H, W]
        pts = np.asarray(d[self.points_key]).copy()  # [N, 5] ndarray (handles MetaTensor/list)
        a, b = self.a, self.b

        size_a = img.shape[a + 1]        # spatial sizes before rotation
        size_b = img.shape[b + 1]

        # Rotate image (dims shifted by 1 for channel axis)
        d[self.image_key] = torch.rot90(img, k=self._k, dims=[a + 1, b + 1])

        # Update point coordinates
        # AA: old sentinel (ZYX: column 0 = z = -100 for padding):
        # valid = pts[:, 0] >= 0
        valid = pts[:, 3] >= 0   # XYZ: padding rows have class_id = -100 at column 3
        old_a = pts[valid, a].copy()
        old_b = pts[valid, b].copy()

        k = self._k % 4
        if k == 1:
            pts[valid, a] = size_b - 1 - old_b
            pts[valid, b] = old_a
        elif k == 2:
            pts[valid, a] = size_a - 1 - old_a
            pts[valid, b] = size_b - 1 - old_b
        elif k == 3:
            pts[valid, a] = old_b
            pts[valid, b] = size_a - 1 - old_a

        d[self.points_key] = pts
        return d


# ---------------------------------------------------------------------------
# Image loader: branches on file type
# ---------------------------------------------------------------------------

def _load_detection_image(x):
    """Load a detection image into a channel-first float tensor [1, D, H, W].

    - .pt  → pre-extracted, z-score normalised training patch (load as-is).
    - .nii / .nii.gz → raw val/test tomogram: load with nibabel (XYZ axis order,
      matching patch generation) and z-score normalise so its intensity scale
      matches the training .pt patches.
    - already-loaded tensor/array → passed through unchanged.
    """
    if not isinstance(x, str):
        return x

    lower = x.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        import nibabel as nib
        vol = nib.load(x).get_fdata().astype(np.float32)        # (X, Y, Z)
        mean, std = vol.mean(), vol.std()
        vol = (vol - mean) / max(float(std), 1e-8)              # nnU-Net-style z-score
        return torch.from_numpy(vol).unsqueeze(0).float()       # [1, X, Y, Z]

    return torch.load(x, map_location="cpu", weights_only=True).unsqueeze(0).float()


# ---------------------------------------------------------------------------
# make_transforms — main entry point
# ---------------------------------------------------------------------------

def make_transforms(crop_size: int = 96):  # noqa: ARG001
    """Return (train_transforms, val_transforms) for 3D detection on CryoET.

    Pretrain-matched augmentation schedule (same as DS001/DS010 in seg Exp 4):
        Geometric (custom, image + points): per-axis flips p=0.3, Rotate90 all planes p=0.3
        Intensity (MONAI, image only)     : AdjustContrast, Gibbs, Scale, Shift, GaussianNoise

    Args:
        crop_size : side length of cubic patch (default 96; patches are pre-extracted)

    Returns:
        (train_transforms, val_transforms) — both are monai.transforms.Compose objects
    """
    _load = [
        # Load image → [1, D, H, W]. Training entries are pre-extracted, z-score
        # normalised .pt patches; val/test entries are raw .nii.gz tomograms that
        # must be loaded with nibabel and z-score normalised here (to match the
        # normalisation baked into the training .pt patches).
        Lambdad(keys=["image"], func=_load_detection_image),
        # JSON datalist stores "points" as a list-of-lists; convert to an Nx5
        # float32 ndarray so the geometric transforms (and collate) can index it.
        # reshape(-1, 5) keeps the right shape even for patches with zero points.
        Lambdad(keys=["points"], func=lambda p: np.asarray(p, dtype=np.float32).reshape(-1, 5)),
        # Per-crop percentile clip to [-1, 1] — matches pretraining ScaleIntensityRangePercentilesd
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False
        ),
    ]

    # Geometric augmentations — ported from Kaggle random_flip_volume / rotate logic
    _geometric = [
        RandFlipWithPointsd(image_key="image", points_key="points", spatial_axis=0, prob=0.3),
        RandFlipWithPointsd(image_key="image", points_key="points", spatial_axis=1, prob=0.3),
        RandFlipWithPointsd(image_key="image", points_key="points", spatial_axis=2, prob=0.3),
        RandRotate90WithPointsd(image_key="image", points_key="points", spatial_axes=(0, 1), prob=0.3),
        RandRotate90WithPointsd(image_key="image", points_key="points", spatial_axes=(1, 2), prob=0.3),
        RandRotate90WithPointsd(image_key="image", points_key="points", spatial_axes=(0, 2), prob=0.3),
    ]

    # Pretrain-matched intensity augmentations — image only, points unchanged
    _intensity = [
        RandAdjustContrastd(keys=["image"], prob=0.8, gamma=(0.5, 2)),
        OneOf([
            RandGaussianSmoothd(keys=["image"], prob=0.1),
            RandGaussianSharpend(keys=["image"], prob=0.1),
        ]),
        RandGibbsNoised(keys=["image"], prob=0.2),
        RandScaleIntensityd(keys=["image"], factors=(1 / 1.1, 1.1), prob=1.0),
        RandShiftIntensityd(keys=["image"], offsets=0.1, safe=False, prob=1.0),
        RandGaussianNoised(keys=["image"], prob=1.0, std=0.002),
        EnsureTyped(keys=["image"]),
    ]

    train_transforms = Compose(_load + _geometric + _intensity)
    val_transforms   = Compose(_load + [EnsureTyped(keys=["image"])])

    return train_transforms, val_transforms
