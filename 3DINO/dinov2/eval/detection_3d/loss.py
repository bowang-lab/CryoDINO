# Ported verbatim from:
#   detection_repo_kaggle/Kaggle-2024-CryoET/cryoet/modelling/detection/task_aligned_assigner.py
#   detection_repo_kaggle/Kaggle-2024-CryoET/cryoet/modelling/detection/functional.py
#
# Minimal changes vs Kaggle:
#   1. pytorch_toolbelt distributed helpers replaced with a lightweight local equivalent.
#   2. convert_2d_to_3d and model-related utilities removed (not needed for loss).
#   3. scores_are_logits flag added to decode_detections_with_nms:
#      sliding_window_accumulate applies sigmoid per-tile before averaging, so
#      we skip the sigmoid in the decode step to avoid double-applying it.
#   4. Print statements inside decode_detections_with_nms removed (too noisy during val).
#   5. Coordinate convention: (X, Y, Z) — same as Kaggle, same as nibabel XYZ axis order
#      (matching pretraining patch generation). GT labels [B, N, 5] = (x, y, z, class, sigma).

from typing import List, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Distributed helpers  (replaces pytorch_toolbelt)
# ---------------------------------------------------------------------------

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def maybe_all_reduce(x: Tensor, op=dist.ReduceOp.SUM) -> Tensor:
    if not is_dist_avail_and_initialized():
        return x
    xc = x.clone()
    dist.all_reduce(xc, op=op)
    return xc


# ---------------------------------------------------------------------------
# TaskAlignedAssigner helpers  (verbatim from task_aligned_assigner.py)
# ---------------------------------------------------------------------------

def batch_pairwise_keypoints_iou(
    pred_keypoints: Tensor,
    true_keypoints: Tensor,
    true_sigmas: Tensor,
) -> Tensor:
    centers1 = pred_keypoints[:, None, :, :]   # [B, 1, M, 3]
    centers2 = true_keypoints[:, :, None, :]   # [B, N, 1, 3]
    d = ((centers1 - centers2) ** 2).sum(dim=-1, keepdim=False)   # [B, N, M]
    sigmas = true_sigmas.reshape(true_keypoints.size(0), true_keypoints.size(1), 1)
    e: Tensor = d / (2 * sigmas ** 2)
    iou = torch.exp(-e)
    return iou


def compute_max_iou_anchor(ious: Tensor) -> Tensor:
    num_max_boxes = ious.shape[-2]
    max_iou_index = ious.argmax(dim=-2)
    is_max_iou: Tensor = F.one_hot(max_iou_index, num_max_boxes).permute([0, 2, 1])
    return is_max_iou.type_as(ious)


def gather_topk_anchors(
    metrics: Tensor, topk: int, largest: bool = True,
    topk_mask: Optional[Tensor] = None, eps: float = 1e-9,
) -> Tensor:
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(dim=-1, keepdim=True).values > eps).type_as(metrics)
    is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(dim=-2).type_as(metrics)
    return is_in_topk * topk_mask


def check_points_inside_bboxes(
    anchor_points: Tensor, gt_centers: Tensor, gt_radius: Tensor, eps: float = 0.05,
) -> Tensor:
    iou = batch_pairwise_keypoints_iou(anchor_points, gt_centers, gt_radius)
    return (iou > eps).type_as(gt_centers)


# ---------------------------------------------------------------------------
# TaskAlignedAssigner  (verbatim from task_aligned_assigner.py)
# ---------------------------------------------------------------------------

class TaskAlignedAssigner(nn.Module):

    def __init__(self, max_anchors_per_point, assigned_min_iou_for_anchor, alpha=1.0, beta=6.0, eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = max_anchors_per_point
        self.assigned_min_iou_for_anchor = assigned_min_iou_for_anchor
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_scores: Tensor,
        pred_centers: Tensor,
        anchor_points: Tensor,
        true_labels: Tensor,
        true_centers: Tensor,
        true_sigmas: Tensor,
        pad_gt_mask: Tensor,
        bg_index: int,
    ):
        assert pred_scores.ndim == pred_centers.ndim
        assert true_labels.ndim == true_centers.ndim and true_centers.ndim == 3

        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = true_centers.shape

        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index, dtype=torch.long, device=true_labels.device)
            assigned_points = torch.zeros([batch_size, num_anchors, 3], device=true_labels.device)
            assigned_scores = torch.zeros([batch_size, num_anchors, num_classes], device=true_labels.device)
            assigned_sigmas = torch.zeros([batch_size, num_anchors], device=true_labels.device)
            return assigned_labels, assigned_points, assigned_scores, assigned_sigmas

        ious = batch_pairwise_keypoints_iou(pred_centers, true_centers, true_sigmas)

        pred_scores = torch.permute(pred_scores, [0, 2, 1])
        batch_ind = torch.arange(end=batch_size, dtype=true_labels.dtype, device=true_labels.device).unsqueeze(-1)
        gt_labels_ind = torch.stack([batch_ind.tile([1, num_max_boxes]), true_labels.squeeze(-1)], dim=-1)

        bbox_cls_scores = pred_scores[gt_labels_ind[..., 0], gt_labels_ind[..., 1]]

        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        is_in_gts = check_points_inside_bboxes(anchor_points, true_centers, true_sigmas, eps=self.assigned_min_iou_for_anchor)

        is_in_topk = gather_topk_anchors(alignment_metrics * is_in_gts, self.topk, topk_mask=pad_gt_mask)

        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile([1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        assigned_gt_index = mask_positive.argmax(dim=-2)

        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = torch.gather(true_labels.flatten(), index=assigned_gt_index.flatten(), dim=0)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(mask_positive_sum > 0, assigned_labels, torch.full_like(assigned_labels, bg_index))

        assigned_points = true_centers.reshape([-1, 3])[assigned_gt_index.flatten(), :]
        assigned_points = assigned_points.reshape([batch_size, num_anchors, 3])

        assigned_sigmas = true_sigmas.reshape([-1])[assigned_gt_index.flatten()]
        assigned_sigmas = assigned_sigmas.reshape([batch_size, num_anchors])

        assigned_scores = F.one_hot(assigned_labels, num_classes + 1)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = torch.index_select(
            assigned_scores, index=torch.tensor(ind, device=assigned_scores.device, dtype=torch.long), dim=-1
        )

        alignment_metrics *= mask_positive
        max_metrics_per_instance = alignment_metrics.max(dim=-1, keepdim=True).values
        max_ious_per_instance = (ious * mask_positive).max(dim=-1, keepdim=True).values
        alignment_metrics = alignment_metrics / (max_metrics_per_instance + self.eps) * max_ious_per_instance
        alignment_metrics = alignment_metrics.max(dim=-2).values.unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        return assigned_labels, assigned_points, assigned_scores, assigned_sigmas


# ---------------------------------------------------------------------------
# Anchor generation  (verbatim from functional.py)
# ---------------------------------------------------------------------------

def anchors_for_offsets_feature_map(offsets, stride):
    z, y, x = torch.meshgrid(
        torch.arange(offsets.size(-3), device=offsets.device),
        torch.arange(offsets.size(-2), device=offsets.device),
        torch.arange(offsets.size(-1), device=offsets.device),
        indexing="ij",
    )
    anchors = torch.stack([x, y, z], dim=0)
    anchors = anchors.float().add_(0.5).mul_(stride)
    anchors = anchors[None, ...].repeat(offsets.size(0), 1, 1, 1, 1)
    return anchors


# ---------------------------------------------------------------------------
# Decode  (verbatim from functional.py, einops kept)
# ---------------------------------------------------------------------------

def decode_detections(
    logits: Union[Tensor, List[Tensor]],
    offsets: Union[Tensor, List[Tensor]],
    strides: Union[int, List[int]],
):
    if torch.is_tensor(logits):
        logits = [logits]
    if torch.is_tensor(offsets):
        offsets = [offsets]
    if isinstance(strides, int):
        strides = [strides]

    anchors = [anchors_for_offsets_feature_map(offset, s) for offset, s in zip(offsets, strides)]

    logits_flat  = []
    centers_flat = []
    anchors_flat = []

    for logit, offset, anchor in zip(logits, offsets, anchors):
        centers = anchor + offset
        logits_flat.append(einops.rearrange(logit,  "B C D H W -> B (D H W) C"))
        centers_flat.append(einops.rearrange(centers, "B C D H W -> B (D H W) C"))
        anchors_flat.append(einops.rearrange(anchor,  "B C D H W -> B (D H W) C"))

    return (
        torch.cat(logits_flat,  dim=1),
        torch.cat(centers_flat, dim=1),
        torch.cat(anchors_flat, dim=1),
    )


# ---------------------------------------------------------------------------
# Loss components  (verbatim from functional.py)
# ---------------------------------------------------------------------------

def varifocal_loss(pred_logits: Tensor, gt_score: Tensor, label: Tensor,
                   alpha=0.75, gamma=2.0) -> Tensor:
    pred_score = pred_logits.sigmoid()
    weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
    loss = weight * F.binary_cross_entropy_with_logits(pred_logits, gt_score, reduction="none")
    return loss.sum()


def keypoint_similarity(pts1, pts2, sigmas):
    d = ((pts1 - pts2) ** 2).sum(dim=-1, keepdim=False)
    e: Tensor = d / (2 * sigmas ** 2)
    return torch.exp(-e)


def iou_loss(pred_centers, assigned_centers, assigned_scores, assigned_sigmas,
             mask_positive, use_l1_loss=False):
    num_pos = mask_positive.sum()
    if num_pos > 0:
        weight = assigned_scores.sum(-1)
        iou = keypoint_similarity(pred_centers, assigned_centers, assigned_sigmas)
        loss = 1 - iou
        if use_l1_loss:
            loss = loss + F.smooth_l1_loss(pred_centers, assigned_centers, reduction="none").sum(-1)
        loss_reduced_iou = torch.masked_fill(loss * weight, ~mask_positive, 0).sum()
        return loss_reduced_iou
    else:
        return torch.zeros([], device=pred_centers.device)


def focal_loss(pred_logits: Tensor, label: Tensor, alpha=0.25, gamma=2.0, reduction="sum") -> Tensor:
    pred_score = pred_logits.sigmoid()
    weight = torch.abs(pred_score - label).pow(gamma)
    if alpha > 0:
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        weight *= alpha_t
    loss = weight * F.binary_cross_entropy_with_logits(pred_logits, label, reduction="none")
    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    return loss


# ---------------------------------------------------------------------------
# Top-level loss  (verbatim from functional.py)
# ---------------------------------------------------------------------------

def object_detection_loss(
    logits:  Union[Tensor, List[Tensor]],
    offsets: Union[Tensor, List[Tensor]],
    strides: Union[int,    List[int]],
    labels:  Tensor,                       # [B, N, 5]  (x, y, z, class, sigma)
    average_tokens_across_devices: bool = False,
    use_l1_loss:          bool = False,
    use_offset_head:      bool = True,
    assigned_min_iou_for_anchor: float = 0.05,
    assigner_max_anchors_per_point: int  = 13,
    assigner_alpha: float = 1.0,
    assigner_beta:  float = 6.0,
    use_varifocal_loss:   bool = True,
    use_cross_entropy_loss: bool = False,
    **kwargs,
) -> Tuple[Tensor, dict]:
    """
    Compute the detection loss adapted for 3D data.

    :param logits:  Predicted heatmap logits  [B, C, D, H, W]
    :param offsets: Predicted offsets         [B, 3, D, H, W]
    :param strides: Voxels per feature cell
    :param labels:  GT tensor [B, N, 5]  cols: (x, y, z, class_id, sigma_vox)
                    Padding rows have class_id == -100.
    """
    pred_logits, pred_centers, anchor_points = decode_detections(logits, offsets, strides)
    batch_size, num_anchors, num_classes = pred_logits.size()

    true_centers = labels[:, :, :3]           # [B, N, 3]
    true_labels  = labels[:, :, 3:4].long()  # [B, N, 1]
    true_sigmas  = labels[:, :, 4:5]         # [B, N, 1]

    assigner = TaskAlignedAssigner(
        max_anchors_per_point=assigner_max_anchors_per_point,
        assigned_min_iou_for_anchor=assigned_min_iou_for_anchor,
        alpha=assigner_alpha,
        beta=assigner_beta,
    )
    assigned_labels, assigned_centers, assigned_scores, assigned_sigmas = assigner(
        pred_scores=pred_logits.detach().sigmoid(),
        pred_centers=pred_centers,
        anchor_points=anchor_points,
        true_labels=torch.masked_fill(true_labels, true_labels.eq(-100), 0),
        true_centers=true_centers,
        true_sigmas=true_sigmas,
        pad_gt_mask=true_labels.ne(-100),
        bg_index=num_classes,
    )

    if use_varifocal_loss:
        one_hot_label = F.one_hot(assigned_labels, num_classes + 1)[..., :-1]
        cls_loss = varifocal_loss(pred_logits, assigned_scores, one_hot_label)
    else:
        cls_loss = focal_loss(pred_logits, assigned_scores, alpha=-1)

    if use_cross_entropy_loss or True:
        bg_label_mask = assigned_labels.eq(num_classes)
        ce_loss = F.cross_entropy(
            input=pred_logits.permute(0, 2, 1),
            target=torch.masked_fill(assigned_labels, bg_label_mask, -100),
            reduction="none",
        )
        cls_loss += torch.sum(ce_loss * assigned_scores.sum(-1))

    if use_offset_head:
        reg_loss = iou_loss(
            pred_centers=pred_centers,
            assigned_centers=assigned_centers,
            assigned_scores=assigned_scores,
            assigned_sigmas=assigned_sigmas,
            mask_positive=assigned_labels != num_classes,
            use_l1_loss=use_l1_loss,
        )
    else:
        reg_loss = torch.tensor(0.0, device=pred_centers.device)

    divisor = assigned_scores.sum()

    if average_tokens_across_devices and is_dist_avail_and_initialized():
        divisor = maybe_all_reduce(divisor.detach())
        cls_loss.mul_(get_world_size())
        reg_loss.mul_(get_world_size())

    divisor = divisor.clamp_min(1)
    cls_loss.div_(divisor)
    reg_loss.div_(divisor)
    loss = cls_loss + reg_loss

    return loss, {
        "loss":               float(loss),
        "cls_loss":           float(cls_loss),
        "reg_loss":           float(reg_loss),
        "num_items_in_batch": float(divisor),
    }


# ---------------------------------------------------------------------------
# Helpers  (verbatim from functional.py)
# ---------------------------------------------------------------------------

def gaussian_blur_3d(x: Tensor, kernel_size: int, sigma: float) -> Tensor:
    kd = kh = kw = kernel_size
    z  = torch.linspace(-(kd // 2), kd // 2, steps=kd)
    y  = torch.linspace(-(kh // 2), kh // 2, steps=kh)
    x_ = torch.linspace(-(kw // 2), kw // 2, steps=kw)
    zz, yy, xx = torch.meshgrid(z, y, x_, indexing="ij")
    kernel_3d = torch.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
    kernel_3d /= kernel_3d.sum()
    kernel_3d = kernel_3d.to(x.device, x.dtype)
    C = x.shape[1]
    kernel_3d = kernel_3d.view(1, 1, kd, kh, kw).repeat(C, 1, 1, 1, 1)
    return F.conv3d(x, weight=kernel_3d, padding=kernel_size // 2, groups=C)


def centernet_heatmap_nms(scores: Tensor, kernel: Union[int, Tuple[int, int, int]] = 3) -> Tensor:
    if isinstance(kernel, int):
        kernel = (kernel, kernel, kernel)
    pad = ((kernel[0] - 1) // 2, (kernel[1] - 1) // 2, (kernel[2] - 1) // 2)
    maxpool = F.max_pool3d(scores, kernel_size=kernel, padding=pad, stride=1)
    return scores * (scores == maxpool)


# ---------------------------------------------------------------------------
# Inference: decode + NMS  (verbatim from functional.py; prints removed;
#                           scores_are_logits flag added for val path)
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_detections_with_nms(
    scores:  List[Tensor],         # each [C, D, H, W]
    offsets: List[Tensor],         # each [3, D, H, W]
    strides: List[int],
    min_score: Union[float, List[float]],
    class_sigmas: List[float],
    iou_threshold: float = 0.25,
    use_single_label_per_anchor: bool = True,
    use_centernet_nms: bool = False,
    pre_nms_top_k: Optional[int] = None,
    class_map_gaussian_smoothing_kernel: int = 0,
    centernet_nms_kernel: Union[int, Tuple[int, int, int]] = 3,
    scores_are_logits: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Decode detections from scores and centers with NMS.

    :param scores:            List of [C, D, H, W] score maps (logits or probas, see scores_are_logits)
    :param offsets:           List of [3, D, H, W] offset maps
    :param min_score:         Score threshold (scalar or per-class list)
    :param class_sigmas:      NMS radius per class in voxels
    :param iou_threshold:     Suppress if Gaussian IOU above this
    :param scores_are_logits: If True (default) apply sigmoid; if False scores are already probas
                              (e.g. from sliding_window_accumulate which averages per-tile probas)
    :return: (final_centers [K, 3] (x,y,z), final_labels [K], final_scores [K])
    """
    num_classes = scores[0].shape[0]

    min_score = np.asarray(min_score, dtype=np.float32).reshape(-1)
    if len(min_score) == 1:
        min_score = np.full(num_classes, min_score[0], dtype=np.float32)

    if class_map_gaussian_smoothing_kernel > 0:
        scores = [
            gaussian_blur_3d(s.unsqueeze(0), kernel_size=class_map_gaussian_smoothing_kernel, sigma=1.0).squeeze(0)
            for s in scores
        ]

    if use_centernet_nms:
        scores = [centernet_heatmap_nms(s.unsqueeze(0), kernel=centernet_nms_kernel).squeeze(0) for s in scores]

    scores, centers, _ = decode_detections(
        [s.unsqueeze(0) for s in scores],
        [o.unsqueeze(0) for o in offsets],
        strides,
    )
    scores  = scores.squeeze(0)   # [L, C]
    centers = centers.squeeze(0)  # [L, 3]

    if scores_are_logits:
        scores = scores.sigmoid()

    labels_of_max_score = scores.argmax(dim=1)

    final_labels_list  = []
    final_scores_list  = []
    final_centers_list = []

    for class_index in range(num_classes):
        sigma_value      = float(class_sigmas[class_index])
        score_threshold  = float(min_score[class_index])
        score_mask       = scores[:, class_index] >= score_threshold

        if use_single_label_per_anchor:
            mask = labels_of_max_score.eq(class_index) & score_mask
        else:
            mask = score_mask

        if not mask.any():
            continue

        class_scores  = scores[mask, class_index]
        class_centers = centers[mask]

        if pre_nms_top_k is not None and len(class_scores) > pre_nms_top_k:
            class_scores, sort_idx = torch.topk(class_scores, pre_nms_top_k, largest=True, sorted=True)
            class_centers = class_centers[sort_idx]
        else:
            class_scores, sort_idx = class_scores.sort(descending=True)
            class_centers = class_centers[sort_idx]

        suppressed  = torch.zeros_like(class_scores, dtype=torch.bool)
        keep_indices = []
        for i in range(class_scores.size(0)):
            if suppressed[i]:
                continue
            keep_indices.append(i)
            iou = keypoint_similarity(class_centers[i:i+1], class_centers, sigma_value)
            suppressed |= (iou > iou_threshold).to(suppressed.device)

        keep_indices = torch.as_tensor(keep_indices, dtype=torch.long, device=class_scores.device)
        final_labels_list.append(torch.full((keep_indices.numel(),), class_index, dtype=torch.long))
        final_scores_list.append(class_scores[keep_indices])
        final_centers_list.append(class_centers[keep_indices])

    if final_centers_list:
        return torch.cat(final_centers_list), torch.cat(final_labels_list), torch.cat(final_scores_list)
    else:
        device = scores[0].device if isinstance(scores, list) else scores.device
        return (
            torch.empty(0, 3, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, device=device),
        )
