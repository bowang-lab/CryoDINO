# Tests for the point-coordinate-aware augmentations.
#
# The docstrings in augmentations.py claim the rotation math is "verified by
# test_augmentations.py" — this is that file. The check is empirical rather than
# algebraic: we plant a unique marker voxel at each point's coordinate, apply the
# transform to image and points jointly, and assert the marker is still exactly at
# the (rounded) transformed coordinate. If the point math and the image math ever
# disagree, that is precisely the silent failure that makes detection untrainable.
#
# Run: pytest dinov2/eval/detection_3d/test_augmentations.py

import numpy as np
import pytest
import torch

from dinov2.eval.detection_3d.augmentations import (
    RandFlipWithPointsd,
    RandRotate90WithPointsd,
)

IMAGE_KEY, POINTS_KEY = "image", "points"


def _volume_with_markers(shape, coords):
    """[1, *shape] volume where voxel i+1 marks coords[i] (x, y, z) = spatial dims (0, 1, 2)."""
    img = torch.zeros((1, *shape), dtype=torch.float32)
    for i, (x, y, z) in enumerate(coords):
        img[0, int(x), int(y), int(z)] = float(i + 1)
    return img


def _marker_location(img, i):
    """Where did marker i+1 end up?"""
    hits = (img[0] == float(i + 1)).nonzero()
    assert hits.shape[0] == 1, f"marker {i + 1} appears {hits.shape[0]} times, expected once"
    return hits[0].tolist()


def _points(coords, sigma=5.0):
    """[N, 5] = (x, y, z, class_id, sigma)."""
    return np.array([[x, y, z, i % 6, sigma] for i, (x, y, z) in enumerate(coords)],
                    dtype=np.float32)


# Deliberately non-cubic, and asymmetric coordinates, so an axis swap cannot pass by luck.
SHAPE = (8, 10, 12)
COORDS = [(1, 2, 3), (7, 0, 11), (4, 9, 5), (0, 5, 0)]


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_flip_moves_points_with_the_image(axis):
    t = RandFlipWithPointsd(IMAGE_KEY, POINTS_KEY, spatial_axis=axis, prob=1.0)
    out = t({IMAGE_KEY: _volume_with_markers(SHAPE, COORDS), POINTS_KEY: _points(COORDS)})

    for i in range(len(COORDS)):
        assert _marker_location(out[IMAGE_KEY], i) == [round(c) for c in out[POINTS_KEY][i, :3]], (
            f"flip axis {axis}: point {i} does not track the image"
        )


@pytest.mark.parametrize("axes", [(0, 1), (1, 2), (0, 2)])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_rot90_moves_points_with_the_image(axes, k):
    t = RandRotate90WithPointsd(IMAGE_KEY, POINTS_KEY, spatial_axes=axes, prob=1.0)
    t._do_transform, t._k = True, k          # pin k instead of sampling it
    t.randomize = lambda data: None

    out = t({IMAGE_KEY: _volume_with_markers(SHAPE, COORDS), POINTS_KEY: _points(COORDS)})

    for i in range(len(COORDS)):
        assert _marker_location(out[IMAGE_KEY], i) == [round(c) for c in out[POINTS_KEY][i, :3]], (
            f"rot90 axes={axes} k={k}: point {i} does not track the image"
        )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_flip_leaves_padding_rows_untouched(axis):
    """Padding rows (class_id = -100 at column 3) must never be transformed — a padded
    coordinate that gets mirrored becomes a plausible-looking in-bounds point."""
    pts = np.concatenate([_points(COORDS), np.full((3, 5), -100.0, dtype=np.float32)])
    t = RandFlipWithPointsd(IMAGE_KEY, POINTS_KEY, spatial_axis=axis, prob=1.0)
    out = t({IMAGE_KEY: _volume_with_markers(SHAPE, COORDS), POINTS_KEY: pts})

    assert np.all(out[POINTS_KEY][len(COORDS):] == -100.0)


def test_rot90_is_identity_after_four_quarter_turns():
    pts = _points(COORDS)
    d = {IMAGE_KEY: _volume_with_markers(SHAPE, COORDS), POINTS_KEY: pts}

    # (0, 1) is the only square plane in SHAPE=(8, 10, 12)? No — use a cubic volume so
    # four turns in any plane return to the original shape.
    cubic = (8, 8, 8)
    coords = [(1, 2, 3), (7, 0, 5), (4, 6, 0)]
    d = {IMAGE_KEY: _volume_with_markers(cubic, coords), POINTS_KEY: _points(coords)}

    for _ in range(4):
        t = RandRotate90WithPointsd(IMAGE_KEY, POINTS_KEY, spatial_axes=(0, 1), prob=1.0)
        t._do_transform, t._k = True, 1
        t.randomize = lambda data: None
        d = t(d)

    np.testing.assert_allclose(d[POINTS_KEY][:, :3], _points(coords)[:, :3], atol=1e-5)




def _scale_multipliers(train_tf, n=50):
    """Multipliers drawn by the single RandScaleIntensityd inside a make_transforms train pipeline."""
    from monai.transforms import RandScaleIntensityd

    scales = [t for t in train_tf.transforms if isinstance(t, RandScaleIntensityd)]
    assert len(scales) == 1
    ones = torch.ones(1, 4, 4, 4)
    out = []
    for seed in range(n):
        scales[0].set_random_state(seed)
        out.append(float(torch.as_tensor(scales[0]({IMAGE_KEY: ones.clone()})[IMAGE_KEY]).mean()))
    return out


def test_intended_scale_factors_keep_eval_range():
    """INTENDED_SCALE_FACTORS must give ~0.9-1.1x, keeping inputs on the [-1, 1] eval-tile scale.

    MONAI's RandScaleIntensity multiplies by (1 + factor), so factors=(1/1.1, 1.1) is ~2x; a
    RetinaNet trained that way loses ~0.3 val F4 when evaluated on [-1, 1] tiles.
    """
    from dinov2.eval.detection_3d.augmentations import INTENDED_SCALE_FACTORS, make_transforms

    train_tf, _ = make_transforms(scale_factors=INTENDED_SCALE_FACTORS)
    for seed, m in enumerate(_scale_multipliers(train_tf)):
        assert 0.85 < m < 1.15, f"seed {seed}: intensity multiplier {m:.3f}, expected ~0.9-1.1"


def test_default_scale_factors_stay_pretrain_matched():
    """The default must stay the pretrain-matched ~2x so CryoDINO fine-tuning (detection3d.py) is unchanged."""
    from dinov2.eval.detection_3d.augmentations import make_transforms

    train_tf, _ = make_transforms()
    for seed, m in enumerate(_scale_multipliers(train_tf)):
        assert 1.85 < m < 2.15, f"seed {seed}: intensity multiplier {m:.3f}, expected pretrain-matched ~2x"
