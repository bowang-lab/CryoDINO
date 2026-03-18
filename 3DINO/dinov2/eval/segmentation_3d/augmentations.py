# Author: Tony Xu
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

# edited by Ahmadreza Attarpour; adapted to CryoET datasets and added new augmentations

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    MapTransform,
    EnsureTyped,
    RandSpatialCropSamplesd,
    RandScaleIntensityd,
    ConcatItemsd,
    DeleteItemsd,
    SpatialPadd,
    Lambdad
)
import torch
import numpy as np


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 3 is GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            result = []
            # merge label 1 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 3 is ET
            result.append(d[key] == 3)
            d[key] = torch.cat(result, dim=0).float()
        return d

## AA CHANGE: add new dataset-specific transforms here
# INSPIRED BY nnUNet's cropping strategy, but adapted to work directly with b2nd arrays for efficient cropping without loading full volumes into memory.
import warnings
import numpy as np
import blosc2
from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple
from monai.transforms import Transform, RandomizableTransform
import os


class ZScoreNormalized(MapTransform):
    """nnUNet-style z-score normalization as a MONAI dict transform.
    Optionally uses segmentation mask to normalize only within foreground.

    Args:
        keys: keys for images to normalize
        label_key: key for segmentation mask (used when use_mask_for_norm=True)
        use_mask_for_norm: if True, compute mean/std only from foreground (seg > 0)
    """
    def __init__(self, keys, label_key=None, use_mask_for_norm=False, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.label_key = label_key
        self.use_mask_for_norm = use_mask_for_norm

    def __call__(self, data):
        d = dict(data)
        seg = d.get(self.label_key) if self.label_key else None

        for key in self.keys:
            image = d[key].float().numpy() if hasattr(d[key], 'numpy') else np.asarray(d[key], dtype=np.float32)

            if self.use_mask_for_norm and seg is not None:
                seg_np = seg.numpy() if hasattr(seg, 'numpy') else np.asarray(seg)
                mask = seg_np > 0
                if mask.any():
                    mean = image[mask].mean()
                    std = image[mask].std()
                    image[mask] = (image[mask] - mean) / max(std, 1e-8)
                else:
                    mean = image.mean()
                    std = image.std()
                    image -= mean
                    image /= max(std, 1e-8)
            else:
                mean = image.mean()
                std = image.std()
                image -= mean
                image /= max(std, 1e-8)

            d[key] = torch.from_numpy(image) if hasattr(d[key], 'numpy') else image
        return d


class LoadB2NDdLazy(Transform):
    """
    Lazy loader for b2nd files - opens with mmap, only loads data on slice access.

    Expects data dict with 'image' and 'label' keys containing paths to .b2nd files.
    The paths should already point to b2nd files (from the _b2nd.json datalist).
    """

    def __init__(self, keys: List[str] = ["image", "label"]):
        self.keys = keys
        self.dparams = {'nthreads': 1}
        self.mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            path = d[key]
            if isinstance(path, str) and path.endswith('.b2nd'):
                # Open b2nd file lazily with mmap - this is instant, no data loaded yet
                d[key] = blosc2.open(
                    urlpath=path,
                    mode='r',
                    dparams=self.dparams,
                    **self.mmap_kwargs
                )
            else:
                raise ValueError(f"Expected .b2nd file path for key '{key}', got: {path}")

        return d


class CropFromB2NDd(RandomizableTransform):
    """
    Crop directly from b2nd arrays using slice operation (only loads cropped region).

    Supports nnUNet-style force_fg cropping that ensures foreground is included.

    Args:
        keys: keys for image and label
        spatial_size: crop size (single int for cubic crop)
        num_samples: number of crops per volume
        force_fg: if True, center crop on a foreground voxel (nnUNet-style)
        fg_classes: list of foreground class values (default: all non-zero)
    """

    def __init__(
        self,
        keys: List[str],
        spatial_size: int,
        num_samples: int = 4,
        force_fg: bool = True,
        fg_classes: Optional[List[int]] = None,
    ):
        self.keys = keys
        self.spatial_size = spatial_size
        self.patch_size = (spatial_size, spatial_size, spatial_size)
        self.num_samples = num_samples
        self.force_fg = force_fg
        self.fg_classes = fg_classes

    def _get_class_locations(self, label: np.ndarray) -> Dict[int, np.ndarray]:
        """Get voxel locations for each foreground class."""
        class_locations = {}

        if self.fg_classes is not None:
            classes = self.fg_classes
        else:
            # All non-zero classes
            classes = [int(c) for c in np.unique(label) if c != 0]

        for c in classes:
            locs = np.argwhere(label == c)
            if len(locs) > 0:
                class_locations[c] = locs

        return class_locations

    def _get_bbox_nnunet(
        self,
        data_shape: Tuple[int, ...],
        class_locations: Dict[int, np.ndarray],
    ) -> Tuple[List[int], List[int]]:
        """
        Get bounding box for crop (nnUNet-style with force_fg).

        Returns:
            bbox_lbs: lower bounds for each dimension
            bbox_ubs: upper bounds for each dimension
        """
        dim = len(data_shape)

        # Calculate padding needed if data is smaller than patch
        need_to_pad = [0] * dim
        for d in range(dim):
            if data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # Valid range for crop start positions
        # lbs can be negative (will need padding), ubs is the max start position
        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i]
               for i in range(dim)]

        if not self.force_fg:
            # Random crop
            bbox_lbs = [np.random.randint(lbs[i], max(lbs[i] + 1, ubs[i] + 1)) for i in range(dim)]
        else:
            # Find eligible classes with voxels
            eligible_classes = [c for c in class_locations.keys() if len(class_locations[c]) > 0]

            if len(eligible_classes) == 0:
                # No foreground found, fall back to random crop
                warnings.warn('No foreground classes found, falling back to random crop')
                bbox_lbs = [np.random.randint(lbs[i], max(lbs[i] + 1, ubs[i] + 1)) for i in range(dim)]
            else:
                # Randomly select a foreground class
                selected_class = eligible_classes[np.random.choice(len(eligible_classes))]

                # Randomly select a voxel from that class
                voxels_of_class = class_locations[selected_class]
                selected_voxel = voxels_of_class[np.random.choice(len(voxels_of_class))]

                # Center crop around selected voxel (with bounds checking)
                bbox_lbs = [
                    max(lbs[i], int(selected_voxel[i]) - self.patch_size[i] // 2)
                    for i in range(dim)
                ]
                # Ensure we don't exceed upper bounds
                bbox_lbs = [
                    min(bbox_lbs[i], max(lbs[i], ubs[i]))
                    for i in range(dim)
                ]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs

    def _crop_with_padding(
        self,
        data: Union[np.ndarray, blosc2.NDArray],
        bbox_lbs: List[int],
        bbox_ubs: List[int],
        pad_value: float = 0,
    ) -> np.ndarray:
        """
        Crop data with padding if bbox extends outside data bounds.

        For b2nd arrays, only loads the valid region from disk.
        """
        is_b2nd = isinstance(data, blosc2.NDArray)
        data_shape = data.shape
        dim = len(bbox_lbs)

        # Calculate actual slice bounds (clamped to data)
        slice_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
        slice_ubs = [min(data_shape[i], bbox_ubs[i]) for i in range(dim)]

        # Build slice tuple
        slices = tuple(slice(slice_lbs[i], slice_ubs[i]) for i in range(dim))

        # Extract the valid region (for b2nd, only loads this region!)
        cropped = data[slices]

        # Convert to numpy if needed
        if is_b2nd:
            cropped = np.asarray(cropped)

        # Calculate padding needed
        pad_before = [slice_lbs[i] - bbox_lbs[i] for i in range(dim)]
        pad_after = [bbox_ubs[i] - slice_ubs[i] for i in range(dim)]

        # Apply padding if needed
        if any(p > 0 for p in pad_before) or any(p > 0 for p in pad_after):
            pad_width = [(pad_before[i], pad_after[i]) for i in range(dim)]
            cropped = np.pad(cropped, pad_width, mode='constant', constant_values=pad_value)

        return cropped

    def __call__(self, data):
        d = dict(data)
        img = d['image']
        lbl = d['label']
        shape = img.shape

        # Load label to find foreground locations (one-time cost per volume)
        # For b2nd, this loads the full label into memory
        if isinstance(lbl, blosc2.NDArray):
            lbl_np = np.asarray(lbl[:])
        else:
            lbl_np = np.asarray(lbl)

        class_locations = self._get_class_locations(lbl_np)

        results = []
        for _ in range(self.num_samples):
            # Get bbox using nnUNet-style force_fg
            bbox_lbs, bbox_ubs = self._get_bbox_nnunet(shape, class_locations)

            # Crop with padding support (for b2nd, only loads cropped region of image!)
            img_crop = self._crop_with_padding(img, bbox_lbs, bbox_ubs, pad_value=-1)
            lbl_crop = self._crop_with_padding(lbl, bbox_lbs, bbox_ubs, pad_value=0)

            # Add channel dimension and convert to torch tensors with consistent dtypes
            # Use float32 for both to match what augmentations expect
            img_crop = torch.from_numpy(img_crop[np.newaxis, ...].copy()).float()
            lbl_crop = torch.from_numpy(lbl_crop[np.newaxis, ...].copy()).float()

            results.append({
                'image': img_crop,
                'label': lbl_crop,
            })

        return results
## AA CHANGE: end of new dataset-specific transforms


class RandInvertedGammad(MapTransform):
    """nnUNet GammaTransform with invert_image=True.
    Normalizes to [0,1], applies 1-(1-x)^gamma, rescales to original range.
    """
    def __init__(self, keys, prob=0.1, gamma_range=(0.7, 1.5), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.prob = prob
        self.gamma_range = gamma_range

    def __call__(self, data):
        d = dict(data)
        if np.random.random() >= self.prob:
            return d
        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        for key in self.keys:
            img = d[key].float()
            img_min, img_max = img.min(), img.max()
            img_range = img_max - img_min
            if img_range < 1e-8:
                continue
            img_norm = (img - img_min) / img_range
            d[key] = (1.0 - (1.0 - img_norm).clamp(0, 1) ** gamma) * img_range + img_min
        return d


class SimulateLowResolutiond(MapTransform):
    """nnUNet SimulateLowResolutionTransform (matched).
    Downsamples with nearest-exact then upsamples back with trilinear.
    Single-channel images: one zoom sampled per call, applied to all spatial dims.
    Applied to image keys only — label is unchanged.

    Args:
        keys: image keys only (label not affected)
        zoom_range: (min, max) zoom factor
        prob: overall probability of applying transform (nnUNet: 0.25)
    """
    def __init__(self, keys, zoom_range=(0.5, 1.0), prob=0.25, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.zoom_range = zoom_range
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        if np.random.random() >= self.prob:
            return d
        zoom = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        for key in self.keys:
            img = d[key].float()       # (C, H, W, D)
            orig_size = img.shape[1:]
            new_shape = [max(1, round(s * zoom)) for s in orig_size]
            downsampled = torch.nn.functional.interpolate(
                img[None], new_shape, mode='nearest-exact'
            )
            up = torch.nn.functional.interpolate(
                downsampled, orig_size, mode='trilinear', align_corners=False
            )[0]
            d[key] = up.to(d[key].dtype)
        return d


def make_transforms(dataset_name, image_size, resize_scale, min_int):

    if dataset_name == 'BTCV':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    elif dataset_name == 'BraTS':

        train_transforms = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image1", "image2", "image3", "image4", "label"], ensure_channel_first=True),
                ConcatItemsd(keys=["image1", "image2", "image3", "image4"], name='image', dim=0),
                DeleteItemsd(keys=["image1", "image2", "image3", "image4"]),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandSpatialCropSamplesd(
                    keys=["image", "label"], num_samples=4, roi_size=(image_size, image_size, image_size), random_size=False
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image1", "image2", "image3", "image4", "label"], ensure_channel_first=True),
                ConcatItemsd(keys=["image1", "image2", "image3", "image4"], name='image', dim=0),
                DeleteItemsd(keys=["image1", "image2", "image3", "image4"]),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    elif dataset_name == 'LA-SEG':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["label"], func=lambda x: (x == 255).astype(np.uint8)),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 0.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["label"], func=lambda x: (x == 255).astype(np.uint8)),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 0.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    elif dataset_name == 'TDSC-ABUS':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    ## AA CHANGE: add new dataset transforms here
    elif "Dataset" in dataset_name:
        from monai.transforms import (
            Compose,
            LoadImaged,
            RandCropByPosNegLabeld,
            EnsureTyped,
            EnsureChannelFirstd,
            RandAdjustContrastd,
            ScaleIntensityRangePercentilesd,
            OneOf,
            RandGaussianSmoothd,
            RandGaussianSharpend,
            RandGibbsNoised,
            RandGaussianNoised,
            RandFlipd,
            RandShiftIntensityd,
            RandHistogramShiftd,
            RandAxisFlipd,
            Transform,
            Lambdad,
            Spacingd,
        )
        from torchio.transforms import RandomAffine
        from skimage.util import random_noise
        class random_salt_pepper(Transform):
            def __call__(self, image_dict):
                # randomly apply salt and pepper noise on 1 percent of the data
                image_dict["image"] = torch.tensor(random_noise(image_dict["image"], mode='s&p', amount=0.001, clip=True))
                return image_dict

        # fdata axis should be H * W * D
        N_crops = 4
        data_aug_prob = 0.7
        crop_size = (image_size, image_size, image_size)

        if "b2nd" in dataset_name:
            # B2ND mode: lazy loading with nnUNet-style force_fg cropping
            # - LoadB2NDdLazy: opens b2nd files with mmap (instant, no data loaded)
            # - CropFromB2NDd: loads only the cropped region from disk + adds channel dim
            # - ScaleIntensityRangePercentilesd applied per-crop to match pretraining (performed worse, reverted)
            train_transforms = Compose(
                [
                    LoadB2NDdLazy(keys=["image", "label"]),
                    CropFromB2NDd(
                        keys=["image", "label"],
                        spatial_size=image_size,
                        num_samples=N_crops,
                        force_fg=True,
                    ),
                    ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False),
                    OneOf(transforms=[
                        RandomAffine(include=["image", "label"], p=data_aug_prob, degrees=(30,30,30),
                                    scales=(0.5, 2), translation=(0.1,0.1,0.1),
                                    default_pad_value='mean'),
                        random_salt_pepper(),
                        RandAdjustContrastd(keys=["image"], prob=data_aug_prob, gamma=(0.5, 4)),
                        RandGaussianSmoothd(keys=["image"], prob=data_aug_prob),
                        RandGaussianNoised(keys=["image"], prob=data_aug_prob, std=0.02),
                        RandHistogramShiftd(keys=["image"], num_control_points=10, prob=data_aug_prob),
                    ]),
                    OneOf([
                        RandScaleIntensityd(keys=["image"], factors=(1/1.1, 1.1), prob=1.0),
                        RandShiftIntensityd(keys=["image"], offsets=0.1, safe=False, prob=1.0)
                    ]),
                    RandAxisFlipd(keys=["image", "label"], prob=data_aug_prob),
                    EnsureTyped(keys=["image", "label"]),
                ]
            )
        elif "patches" in dataset_name:
            Pos_ratio = 3
            if "10001" in dataset_name:
                N_crops = 3
            else:
                N_crops = 4
            # Patches are pre-extracted .pt files (zscore normalized), 4 crops of 112^3 per patch
            load_transforms = [
                Lambdad(keys=["image"], func=lambda x: torch.load(x, map_location='cpu', weights_only=True).unsqueeze(0).float()),
                Lambdad(keys=["label"], func=lambda x: torch.load(x, map_location='cpu', weights_only=True).unsqueeze(0).float()),
            ]
            # Binary label merging for Dataset010 (all foreground classes → 1)
            if "10010" in dataset_name:
                load_transforms.append(Lambdad(keys=["label"], func=lambda x: (x > 0).float()))
            # Dataset049 (12049): merge classes 1,5 into bg; remap 2→1, 3→2, 4→3
            # if "12049" in dataset_name:
            #     def _remap_12049(x):
            #         out = torch.zeros_like(x)
            #         out[x == 2] = 1
            #         out[x == 3] = 2
            #         out[x == 4] = 3
            #         return out.float()
            #     load_transforms.append(Lambdad(keys=["label"], func=_remap_12049))
            # train_transforms = Compose(
            #     load_transforms + [
            #         RandCropByPosNegLabeld(
            #             keys=["image", "label"], label_key="label",
            #             spatial_size=crop_size, pos=5, neg=1,
            #             num_samples=N_crops, image_key="image", image_threshold=-1,
            #         ),
            #         ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False),
            #         OneOf(transforms=[
            #             RandomAffine(include=["image", "label"], p=data_aug_prob, degrees=(30,30,30),
            #                         scales=(0.5, 2), translation=(0.1,0.1,0.1),
            #                         default_pad_value='mean'),
            #             random_salt_pepper(),
            #             RandAdjustContrastd(keys=["image"], prob=data_aug_prob, gamma=(0.5, 4)),
            #             RandGaussianSmoothd(keys=["image"], prob=data_aug_prob),
            #             RandGaussianNoised(keys=["image"], prob=data_aug_prob, std=0.02),
            #             RandHistogramShiftd(keys=["image"], num_control_points=10, prob=data_aug_prob),
            #         ]),
            #         OneOf([
            #             RandScaleIntensityd(keys=["image"], factors=(1/1.1, 1.1), prob=1.0),
            #             RandShiftIntensityd(keys=["image"], offsets=0.1, safe=False, prob=1.0)
            #         ]),
            #         RandAxisFlipd(keys=["image", "label"], prob=data_aug_prob),
            #         EnsureTyped(keys=["image", "label"]),
            #     ]
            # )

            ## AA EXPERIMENT: pretraining-matched augmentations
            # Geometric: per-axis flips (prob=0.3) + 90° rotations on all 3 planes (prob=0.3) — no affine/scale deformation
            # Intensity: sequential RandAdjustContrast → RandGaussianSmooth → RandScaleIntensity → RandShiftIntensity → RandGaussianNoise (same params as pretraining)
            # train_transforms = Compose(
            #     load_transforms + [
            #         RandCropByPosNegLabeld(
            #             keys=["image", "label"], label_key="label",
            #             spatial_size=crop_size, pos=Pos_ratio, neg=1,
            #             num_samples=N_crops, image_key="image", image_threshold=-1,
            #         ),
            #         ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False),
            #         RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.3),
            #         RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.3),
            #         RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.3),
            #         RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 1)),
            #         RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(1, 2)),
            #         RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 2)),
            #         RandAdjustContrastd(keys=["image"], prob=0.8, gamma=(0.5, 2)),
            #         OneOf([
            #             RandGaussianSmoothd(keys=["image"], prob=0.1),
            #             RandGaussianSharpend(keys=["image"], prob=0.1),
            #         ]),
            #         RandGibbsNoised(keys=["image"], prob=0.2),
            #         RandScaleIntensityd(keys=["image"], factors=(1/1.1, 1.1), prob=1.0),
            #         RandShiftIntensityd(keys=["image"], offsets=0.1, safe=False, prob=1.0),
            #         RandGaussianNoised(keys=["image"], prob=1.0, std=0.002),
            #         EnsureTyped(keys=["image", "label"]),
            #     ]
            # )
            ## AA EXPERIMENT END

            ## AA EXPERIMENT 2: nnU-Net-matched augmentations (order matches nnUNet pipeline)
            # Order follows nnUNet 3D full-res augmentation pipeline:
            #  1. SpatialTransform   : RandomAffine (p=0.2, ±30°, scale 0.7–1.4); elastic DISABLED
            #  2. GaussianNoise      : p=0.1, std=0.1
            #  3. GaussianBlur       : p=0.2, σ=0.5–1.0
            #  4. BrightnessMult     : p=0.15, ×0.75–1.25
            #  5. ContrastTransform  : p=0.15, γ=0.75–1.25
            #  6. SimulateLowRes     : p=0.25, zoom=0.5–1.0
            #  7. GammaTransform inv : p=0.1,  γ=0.7–1.5
            #  8. GammaTransform     : p=0.3,  γ=0.7–1.5
            #  9. MirrorTransform    : p=0.5 per axis
            # Extra (pretraining-matched, after nnUNet pipeline):
            #     RandRotate90 (p=0.3, all planes), GibbsNoise (p=0.2), ShiftIntensity (p=0.15)
            # train_transforms = Compose(
            #     load_transforms + [
            #         RandCropByPosNegLabeld(
            #             keys=["image", "label"], label_key="label",
            #             spatial_size=crop_size, pos=Pos_ratio, neg=1,
            #             num_samples=N_crops, image_key="image", image_threshold=-1,
            #         ),
            #         ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False),
            #         # 1. SpatialTransform (rotation p=0.2 ±30°, scale p=0.2 0.7–1.4; elastic disabled for 3D full-res)
            #         RandomAffine(include=["image", "label"], p=0.2,
            #                      degrees=(30, 30, 30),
            #                      scales=(0.7, 1.4),
            #                      default_pad_value='mean'),
            #         # 2. GaussianNoise
            #         RandGaussianNoised(keys=["image"], prob=0.1, std=0.1),
            #         # 3. GaussianBlur
            #         RandGaussianSmoothd(keys=["image"], prob=0.2,
            #                             sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
            #         # 4. MultiplicativeBrightness
            #         RandScaleIntensityd(keys=["image"], factors=0.25, prob=0.15),
            #         # 5. ContrastTransform
            #         RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.75, 1.25)),
            #         # 6. SimulateLowResolution
            #         SimulateLowResolutiond(keys=["image"], zoom_range=(0.5, 1.0), prob=0.25),
            #         # 7. GammaTransform (inverted)
            #         RandInvertedGammad(keys=["image"], prob=0.1, gamma_range=(0.7, 1.5)),
            #         # 8. GammaTransform (regular)
            #         RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            #         # 9. MirrorTransform
            #         RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
            #         RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
            #         RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
            #         # Extra: pretraining-matched
            #         RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 1)),
            #         RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(1, 2)),
            #         RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 2)),
            #         RandGibbsNoised(keys=["image"], prob=0.2),
            #         RandShiftIntensityd(keys=["image"], offsets=0.1, safe=False, prob=0.15),
            #         EnsureTyped(keys=["image", "label"]),
            #     ]
            # )
            ## AA EXPERIMENT 2 END

            ## AA EXPERIMENT 3: pretrain-matched augmentations + nnUNet-only additions
            # Keeps the full pretrain-matched pipeline unchanged, adds three transforms
            # that were completely absent from pretrain-matched but present in nnUNet:
            #   + RandomAffine    : continuous rotation ±30°, scale 0.7–1.4 (p=0.2)
            #   + SimulateLowRes  : nearest-exact downsample → trilinear upsample (p=0.25)
            #   + InvertedGamma   : γ=0.7–1.5 on inverted image (p=0.1)
            train_transforms = Compose(
                load_transforms + [
                    RandCropByPosNegLabeld(
                        keys=["image", "label"], label_key="label",
                        spatial_size=crop_size, pos=Pos_ratio, neg=1,
                        num_samples=N_crops, image_key="image", image_threshold=-1,
                    ),
                    ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False),
                    # nnUNet addition: continuous spatial transform
                    RandomAffine(include=["image", "label"], p=0.2,
                                 degrees=(30, 30, 30),
                                 scales=(0.7, 1.4),
                                 default_pad_value='mean'),
                    # pretrain-matched geometric
                    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.3),
                    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.3),
                    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.3),
                    RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 1)),
                    RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(1, 2)),
                    RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 2)),
                    # pretrain-matched intensity (unchanged)
                    RandAdjustContrastd(keys=["image"], prob=0.8, gamma=(0.5, 2)),
                    OneOf([
                        RandGaussianSmoothd(keys=["image"], prob=0.1),
                        RandGaussianSharpend(keys=["image"], prob=0.1),
                    ]),
                    RandGibbsNoised(keys=["image"], prob=0.2),
                    RandScaleIntensityd(keys=["image"], factors=(1/1.1, 1.1), prob=1.0),
                    RandShiftIntensityd(keys=["image"], offsets=0.1, safe=False, prob=1.0),
                    RandGaussianNoised(keys=["image"], prob=1.0, std=0.002),
                    # nnUNet additions: resolution degradation + inverted gamma
                    SimulateLowResolutiond(keys=["image"], zoom_range=(0.5, 1.0), prob=0.25),
                    RandInvertedGammad(keys=["image"], prob=0.1, gamma_range=(0.7, 1.5)),
                    EnsureTyped(keys=["image", "label"]),
                ]
            )
            ## AA EXPERIMENT 3 END
        else:
            # NIfTI mode: standard MONAI loading with tomogram-level intensity normalization
            # Note: per-crop normalization (after cropping) was tested but performed worse, so keeping tomogram-level
            nifti_load_transforms = [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
            ]
            # Binary label merging for Dataset010 (all foreground classes → 1)
            if "10010" in dataset_name:
                nifti_load_transforms.append(Lambdad(keys=["label"], func=lambda x: (x > 0).float()))
            # Dataset049 (12049): merge classes 1,5 into bg; remap 2→1, 3→2, 4→3
            # if "12049" in dataset_name:
            #     def _remap_12049(x):
            #         out = torch.zeros_like(x)
            #         out[x == 2] = 1
            #         out[x == 3] = 2
            #         out[x == 4] = 3
            #         return out.float()
            #     nifti_load_transforms.append(Lambdad(keys=["label"], func=_remap_12049))
            train_transforms = Compose(
                nifti_load_transforms + [
                    ZScoreNormalized(keys=["image"]),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"], label_key="label",
                        spatial_size=crop_size, pos=1, neg=1,
                        num_samples=N_crops, image_key="image", image_threshold=-1,
                    ),
                    ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False),
                    OneOf(transforms=[
                        RandomAffine(include=["image", "label"], p=data_aug_prob, degrees=(30,30,30),
                                    scales=(0.5, 2), translation=(0.1,0.1,0.1),
                                    default_pad_value='mean'),
                        random_salt_pepper(),
                        RandAdjustContrastd(keys=["image"], prob=data_aug_prob, gamma=(0.5, 4)),
                        RandGaussianSmoothd(keys=["image"], prob=data_aug_prob),
                        RandGaussianNoised(keys=["image"], prob=data_aug_prob, std=0.02),
                        RandHistogramShiftd(keys=["image"], num_control_points=10, prob=data_aug_prob),
                    ]),
                    OneOf([
                        RandScaleIntensityd(keys=["image"], factors=(1/1.1, 1.1), prob=1.0),
                        RandShiftIntensityd(keys=["image"], offsets=0.1, safe=False, prob=1.0)
                    ]),
                    RandAxisFlipd(keys=["image", "label"], prob=data_aug_prob),
                    EnsureTyped(keys=["image", "label"]),
                ]
            )

        # Validation uses NIfTI loading (full volume needed for sliding window inference)
        val_load_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
        ]
        # Binary label merging for Dataset010 (all foreground classes → 1)
        if "10010" in dataset_name:
            val_load_transforms.append(Lambdad(keys=["label"], func=lambda x: (x > 0).float()))
        # Dataset049 (12049): merge classes 1,5 into bg; remap 2→1, 3→2, 4→3
        # if "12049" in dataset_name:
        #     def _remap_12049(x):
        #         out = torch.zeros_like(x)
        #         out[x == 2] = 1
        #         out[x == 3] = 2
        #         out[x == 4] = 3
        #         return out.float()
        #     val_load_transforms.append(Lambdad(keys=["label"], func=_remap_12049))
        val_transforms = Compose(
            val_load_transforms + [
                # ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False),
                ZScoreNormalized(keys=["image"]),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # Flatten transforms to allow caching of non-random transforms if needed
    train_transforms = train_transforms.flatten()
    val_transforms = val_transforms.flatten()

    return train_transforms, val_transforms
