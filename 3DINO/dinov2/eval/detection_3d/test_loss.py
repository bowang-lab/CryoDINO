# Tests for object_detection_loss / TaskAlignedAssigner / decode_detections.
#
# These pin the properties that, when they break, make detection silently
# untrainable rather than loudly wrong:
#   - the assigner actually produces positives for real GT,
#   - padding rows (class_id = -100) never become positives,
#   - an ideal prediction drives the loss to ~0 (the targets are reachable),
#   - the loss is descendable by gradient descent,
#   - anchors/offsets/GT all agree on the same coordinate space.
#
# Run: pytest dinov2/eval/detection_3d/test_loss.py

import numpy as np
import pytest
import torch

from dinov2.eval.detection_3d.loss import (
    TaskAlignedAssigner,
    anchors_for_offsets_feature_map,
    decode_detections,
    object_detection_loss,
)

STRIDE = 2
NUM_CLASSES = 6
GRID = 16                      # feature cells per axis -> patch is GRID * STRIDE voxels
EXTENT = GRID * STRIDE
BG_BIAS = -4.0                 # detection_heads.py initializes cls_head.bias to -4


def _labels(rows, batch=1, pad_to=None):
    """[B, N, 5] = (x, y, z, class_id, sigma), padded exactly as detection_collate_fn does."""
    n = pad_to or max(len(r) for r in rows)
    out = torch.full((batch, n, 5), -100.0)
    for b, r in enumerate(rows):
        if r:
            out[b, :len(r)] = torch.tensor(r, dtype=torch.float32)
    return out


def _init_maps(batch=1):
    """What the head emits at initialization: uniform -4 logits, zero offsets."""
    cls_map = torch.full((batch, NUM_CLASSES, GRID, GRID, GRID), BG_BIAS)
    off_map = torch.zeros((batch, 3, GRID, GRID, GRID))
    return cls_map, off_map


def _assign(labels, cls_map, off_map):
    """Run the assigner exactly as object_detection_loss does, and return its output."""
    pred_logits, pred_centers, anchor_points = decode_detections(cls_map, off_map, STRIDE)
    true_labels = labels[:, :, 3:4].long()
    assigner = TaskAlignedAssigner(max_anchors_per_point=13, assigned_min_iou_for_anchor=0.05,
                                   alpha=1.0, beta=6.0)
    return assigner(
        pred_scores=pred_logits.detach().sigmoid(),
        pred_centers=pred_centers,
        anchor_points=anchor_points,
        true_labels=torch.masked_fill(true_labels, true_labels.eq(-100), 0),
        true_centers=labels[:, :, :3],
        true_sigmas=labels[:, :, 4:5],
        pad_gt_mask=true_labels.ne(-100),
        bg_index=NUM_CLASSES,
    )


# ---------------------------------------------------------------------------
# coordinate space
# ---------------------------------------------------------------------------

def test_anchors_are_in_input_voxel_units():
    """Anchors must span the input patch, not the feature grid — GT arrives in voxels."""
    anchors = anchors_for_offsets_feature_map(torch.zeros(1, 3, GRID, GRID, GRID), STRIDE)
    assert anchors.shape == (1, 3, GRID, GRID, GRID)
    assert float(anchors.min()) == pytest.approx(0.5 * STRIDE)
    assert float(anchors.max()) == pytest.approx((GRID - 1 + 0.5) * STRIDE)


def test_each_offset_channel_moves_its_own_axis():
    """Channel k of off_map must displace coordinate k — an axis swap here is invisible
    in the loss value but makes every predicted center wrong."""
    for axis in range(3):
        off_map = torch.zeros(1, 3, GRID, GRID, GRID)
        off_map[0, axis] = 1.0
        _, centers, anchors = decode_detections(torch.zeros(1, NUM_CLASSES, GRID, GRID, GRID),
                                                off_map, STRIDE)
        delta = (centers - anchors)[0].mean(dim=0)
        expected = torch.zeros(3)
        expected[axis] = 1.0
        torch.testing.assert_close(delta, expected)


# ---------------------------------------------------------------------------
# the assigner
# ---------------------------------------------------------------------------

def test_assigner_produces_positives_for_real_gt():
    """The decisive sanity check: if this is 0, nothing can ever train."""
    labels = _labels([[[15.0, 17.0, 13.0, 3.0, 6.0]]])
    assigned_labels, _, _, _ = _assign(labels, *_init_maps())
    n_pos = int((assigned_labels != NUM_CLASSES).sum())
    assert n_pos == 13, f"expected topk=13 positives for one GT, got {n_pos}"


def test_assigned_positives_carry_the_correct_class():
    labels = _labels([[[15.0, 17.0, 13.0, 3.0, 6.0]]])
    assigned_labels, _, _, _ = _assign(labels, *_init_maps())
    pos = assigned_labels[assigned_labels != NUM_CLASSES]
    assert torch.all(pos == 3)


def test_positives_are_the_anchors_nearest_the_gt():
    """Positives must be geometrically adjacent to the GT; at init the alignment metric is
    iou**6, so the topk should be the nearest anchors."""
    gt = torch.tensor([15.0, 17.0, 13.0])
    labels = _labels([[[*gt.tolist(), 3.0, 6.0]]])
    assigned_labels, _, _, _ = _assign(labels, *_init_maps())

    _, _, anchors = decode_detections(*_init_maps(), STRIDE)
    pos = (assigned_labels != NUM_CLASSES)[0]
    assert (anchors[0][pos] - gt).abs().max() <= 3 * STRIDE


def test_padding_rows_never_become_positives():
    """Padded rows are (-100, -100, -100, -100, -100), so sigma**2 = 1e4 gives them a
    non-negligible Gaussian IoU. They must be masked out of the assignment entirely —
    a padded winner would inject a target center of -100."""
    labels = _labels([[[15.0, 17.0, 13.0, 3.0, 6.0]]], pad_to=40)
    assigned_labels, assigned_centers, _, assigned_sigmas = _assign(labels, *_init_maps())

    pos = assigned_labels != NUM_CLASSES
    assert int(pos.sum()) == 13
    assert float(assigned_centers[pos].min()) >= 0.0, "a padding row won the assignment"
    assert float(assigned_sigmas[pos].min()) > 0.0


def test_batch_with_no_gt_produces_no_positives():
    labels = _labels([[]], pad_to=1)
    assigned_labels, _, _, _ = _assign(labels, *_init_maps())
    assert int((assigned_labels != NUM_CLASSES).sum()) == 0


# ---------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------

def _oracle_maps(labels):
    """The ideal prediction: confident correct class and the exact residual at each GT's
    nearest anchor. The residual must fit the head's tanh()*2 output range."""
    b = labels.shape[0]
    cls_map = torch.full((b, NUM_CLASSES, GRID, GRID, GRID), -10.0)
    off_map = torch.zeros((b, 3, GRID, GRID, GRID))
    for i in range(b):
        for row in labels[i]:
            if row[3] < 0:
                continue
            idx = torch.clamp((row[:3] / STRIDE).floor().long(), 0, GRID - 1)
            anchor = (idx.float() + 0.5) * STRIDE
            cls_map[i, int(row[3]), idx[0], idx[1], idx[2]] = 10.0
            off_map[i, :, idx[0], idx[1], idx[2]] = row[:3] - anchor
    assert float(off_map.abs().max()) <= 2.0, "residual exceeds the head's tanh()*2 range"
    return cls_map, off_map


def test_ideal_prediction_drives_loss_to_zero():
    """If this fails the targets are unreachable and no amount of training will help."""
    labels = _labels([[[15.0, 17.0, 13.0, 3.0, 6.0], [5.0, 25.0, 9.0, 1.0, 8.0]]])
    loss, d = object_detection_loss(*_oracle_maps(labels), strides=STRIDE, labels=labels)
    assert d["loss"] < 0.05, f"ideal prediction still costs {d['loss']:.4f}"
    assert d["reg_loss"] < 0.01


def test_loss_is_finite_with_and_without_padding():
    rows = [[15.0, 17.0, 13.0, 3.0, 6.0]]
    for pad in (None, 40):
        labels = _labels([rows], pad_to=pad)
        loss, d = object_detection_loss(*_init_maps(), strides=STRIDE, labels=labels)
        assert np.isfinite(d["loss"]) and np.isfinite(d["cls_loss"]) and np.isfinite(d["reg_loss"])


def test_loss_is_finite_when_batch_has_no_gt():
    labels = _labels([[]], pad_to=1)
    loss, d = object_detection_loss(*_init_maps(), strides=STRIDE, labels=labels)
    assert np.isfinite(d["loss"])
    assert d["reg_loss"] == 0.0


def test_loss_descends_under_gradient_descent():
    """Optimize the maps directly as free parameters. This isolates the loss from the
    network: if the loss cannot be descended here, the bug is in the loss."""
    labels = _labels([[[15.0, 17.0, 13.0, 3.0, 6.0], [5.0, 25.0, 9.0, 1.0, 8.0]]])
    cls_map, off_map = _init_maps()
    cls_map, off_map = cls_map.requires_grad_(), off_map.requires_grad_()
    opt = torch.optim.AdamW([cls_map, off_map], lr=0.05)

    first = None
    for _ in range(150):
        loss, d = object_detection_loss(cls_map, off_map.tanh() * 2, strides=STRIDE, labels=labels)
        first = first if first is not None else d["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert d["loss"] < 0.25 * first, f"loss only moved {first:.3f} -> {d['loss']:.3f}"


def test_cross_entropy_term_respects_its_flag():
    """loss.py read `if use_cross_entropy_loss or True:`, so the documented default
    (False) silently did nothing and a softmax CE was always added on top of the
    varifocal loss. Toggling the flag must change the loss."""
    labels = _labels([[[15.0, 17.0, 13.0, 3.0, 6.0]]])
    maps = _init_maps()
    off = object_detection_loss(*maps, strides=STRIDE, labels=labels, use_cross_entropy_loss=False)[1]
    on = object_detection_loss(*maps, strides=STRIDE, labels=labels, use_cross_entropy_loss=True)[1]
    assert on["cls_loss"] > off["cls_loss"], "use_cross_entropy_loss has no effect"
