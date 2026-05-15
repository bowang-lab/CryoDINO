# Author: Ahmadreza Attarpour
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.
#
# Adapted from:
#   detection_repo_kaggle/Kaggle-2024-CryoET/cryoet/metric.py
#
# Adaptations vs Kaggle original:
#   1. compute_metrics() copied verbatim (no changes).
#   2. score_submission() (DataFrame-based) replaced by class-based
#      CZIDetectionMetrics / BYUDetectionMetrics with accumulate() /
#      threshold_sweep() interface that plugs directly into our training loop.
#   3. Particle radii and weights kept identical to Kaggle metric.py.

import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Copied verbatim from Kaggle cryoet/metric.py
# ---------------------------------------------------------------------------

def compute_metrics(reference_points, reference_radius, candidate_points):
    """KDTree radius search returning (tp, fp, fn).

    A GT particle is TP if at least one prediction lands within reference_radius.
    Each GT particle counts at most once (duplicates removed via set).
    Copied verbatim from Kaggle cryoet/metric.py.
    """
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0
    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree       = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches    = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)

    matches_within_threshold = set()
    for match in raw_matches:
        matches_within_threshold.update(match)

    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# CZI Detection Metrics  (6 particle classes, Kaggle CZII competition)
# ---------------------------------------------------------------------------

class CZIDetectionMetrics:
    """F4 metric for CZI 6-class particle detection.

    Particle radii and per-class weights are identical to Kaggle cryoet/metric.py.

    Class IDs (0-indexed, matching our detection head output):
        0  ferritin complex      radius=60 Å   weight=1
        1  Beta-amylase          radius=65 Å   weight=0  (excluded from score)
        2  Beta-galactosidase    radius=90 Å   weight=2
        3  cytosolic ribosome    radius=150 Å  weight=1
        4  Thyroglobulin         radius=130 Å  weight=2
        5  virus-like capsid     radius=135 Å  weight=1

    Usage:
        metric = CZIDetectionMetrics()
        for tomo in val_loader:
            pred_centers, pred_labels, pred_scores = decode_detections_with_nms(...)
            metric.accumulate(pred_centers, pred_labels, pred_scores, gt_points, voxel_size)
        best_f4, best_thr, per_class_f4 = metric.threshold_sweep()
        metric.reset()
    """

    PARTICLE_NAMES     = ['ferritin complex', 'Beta-amylase', 'Beta-galactosidase',
                          'cytosolic ribosome', 'Thyroglobulin', 'virus-like capsid']
    PARTICLE_RADII_ANG = [60, 65, 90, 150, 130, 135]
    WEIGHTS            = [1,  0,  2,  1,   2,   1]

    def __init__(self, distance_multiplier: float = 0.5, beta: int = 4):
        self.distance_multiplier = distance_multiplier
        self.beta  = beta
        self._data = []

    def accumulate(
        self,
        pred_centers_vox,   # [K, 3] (x, y, z) in voxels — numpy or torch
        pred_labels,         # [K]    0-indexed class ids
        pred_scores,         # [K]    confidence scores in [0, 1]
        gt_points,           # [N, 5] (x, y, z, class_id, sigma_vox) — padding rows class_id=-100
        voxel_size: float,   # Å per voxel
    ):
        """Store raw per-tomogram predictions for later threshold sweep.

        Padding rows (class_id == -100) are filtered out automatically.
        Torch tensors are converted to numpy.
        """
        if hasattr(pred_centers_vox, 'cpu'):
            pred_centers_vox = pred_centers_vox.cpu().numpy()
        if hasattr(pred_labels, 'cpu'):
            pred_labels = pred_labels.cpu().numpy()
        if hasattr(pred_scores, 'cpu'):
            pred_scores = pred_scores.cpu().numpy()
        if hasattr(gt_points, 'cpu'):
            gt_points = gt_points.cpu().numpy()

        # filter padding rows
        valid = gt_points[:, 3] >= 0
        gt_points = gt_points[valid]

        self._data.append({
            'pred_centers_ang': pred_centers_vox * voxel_size,    # [K, 3] Å
            'pred_labels':      pred_labels.astype(int),           # [K]
            'pred_scores':      pred_scores.astype(np.float32),    # [K]
            'gt_centers_ang':   gt_points[:, :3] * voxel_size,    # [N, 3] Å
            'gt_labels':        gt_points[:, 3].astype(int),       # [N]
        })

    def _f4_at_threshold(self, threshold: float):
        """Compute aggregated F4 and per-class F4 at a given score threshold."""
        total_tp = np.zeros(len(self.PARTICLE_RADII_ANG))
        total_fp = np.zeros(len(self.PARTICLE_RADII_ANG))
        total_fn = np.zeros(len(self.PARTICLE_RADII_ANG))

        for d in self._data:
            keep         = d['pred_scores'] >= threshold
            pred_centers = d['pred_centers_ang'][keep]
            pred_labels  = d['pred_labels'][keep]
            gt_centers   = d['gt_centers_ang']
            gt_labels    = d['gt_labels']

            for cls_idx, radius_ang in enumerate(self.PARTICLE_RADII_ANG):
                ref_radius = radius_ang * self.distance_multiplier
                ref_pts    = gt_centers[gt_labels == cls_idx]
                cand_pts   = pred_centers[pred_labels == cls_idx]
                tp, fp, fn = compute_metrics(ref_pts, ref_radius, cand_pts)
                total_tp[cls_idx] += tp
                total_fp[cls_idx] += fp
                total_fn[cls_idx] += fn

        beta = self.beta
        per_class_f4 = {}
        weighted_sum = 0.0
        total_weight = sum(self.WEIGHTS)

        for cls_idx, (name, weight) in enumerate(zip(self.PARTICLE_NAMES, self.WEIGHTS)):
            tp, fp, fn = total_tp[cls_idx], total_fp[cls_idx], total_fn[cls_idx]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f4   = ((1 + beta**2) * prec * rec / (beta**2 * prec + rec)
                    if (prec + rec) > 0 else 0.0)
            per_class_f4[name] = f4
            weighted_sum += f4 * weight

        agg_f4 = weighted_sum / total_weight if total_weight > 0 else 0.0
        return agg_f4, per_class_f4

    def threshold_sweep(self, n_thresholds: int = 20):
        """Sweep score threshold per class independently, return aggregate F4.

        Mirrors Kaggle's validation metric: each class gets its own optimal threshold,
        then the aggregate weighted F4 is computed from per-class bests. This gives a
        higher (more optimistic) F4 than a single global threshold but matches the
        competition evaluation and correctly ranks checkpoints.

        Returns:
            best_f4         : float — weighted aggregate from per-class best thresholds
            best_thresholds : dict {particle_name: best_threshold_for_that_class}
            best_per_cls_f4 : dict {particle_name: best_f4_for_that_class}
        """
        # AA: old single-global-threshold sweep:
        # best_f4, best_thr, best_per_class = -1.0, float(thresholds[0]), {}
        # for thr in thresholds:
        #     f4, per_class = self._f4_at_threshold(float(thr))
        #     if f4 > best_f4:
        #         best_f4, best_thr, best_per_class = f4, float(thr), per_class
        # return best_f4, best_thr, best_per_class

        thresholds = np.linspace(0.05, 1.0, n_thresholds) ** 2

        best_cls_f4  = {name: 0.0               for name in self.PARTICLE_NAMES}
        best_cls_thr = {name: float(thresholds[0]) for name in self.PARTICLE_NAMES}

        for thr in thresholds:
            _, per_class = self._f4_at_threshold(float(thr))
            for name, f4 in per_class.items():
                if f4 > best_cls_f4[name]:
                    best_cls_f4[name]  = f4
                    best_cls_thr[name] = float(thr)

        weighted_sum = sum(
            best_cls_f4[name] * w
            for name, w in zip(self.PARTICLE_NAMES, self.WEIGHTS)
        )
        total_weight = sum(self.WEIGHTS)
        best_f4 = weighted_sum / total_weight if total_weight > 0 else 0.0

        return best_f4, best_cls_thr, best_cls_f4

    def reset(self):
        self._data = []


# ---------------------------------------------------------------------------
# BYU Detection Metrics  (bacterial motor detection, multi-instance)
# ---------------------------------------------------------------------------

class BYUDetectionMetrics:
    """F-beta metric for BYU bacterial motor detection.

    Adapted from BYU competition metric (distance_metric + fbeta_score functions).
    Key facts from train_labels.csv:
      - Coordinates: (axis 0, axis 1, axis 2) in voxels — axis 0 is depth.
      - Voxel spacing varies per tomogram (6.5–19.7 Å/vox).
      - Some tomograms have 0 motors, some have 1, some have multiple (up to 6+).
      - Distance threshold in voxels = min_radius / voxel_size  (varies per tomogram).

    Adaptation vs BYU competition metric:
      - BYU official: binary y_true/y_pred per GT-instance row, sklearn.fbeta_score.
      - Here: aggregated TP/FP/FN via compute_metrics per tomogram, manual F-beta.
        This gives the same relative ranking of checkpoints.
      - threshold_sweep sweeps confidence score threshold (not distance threshold).

    Args:
        min_radius : matching distance threshold in Å.
                     TODO: verify exact value from BYU competition page.
        beta       : F-beta exponent. TODO: verify (likely 2).
    """

    PARTICLE_NAMES = ['motor']

    def __init__(self, min_radius: float = 500.0, beta: float = 2.0):
        self.min_radius = min_radius
        self.beta       = beta
        self._data      = []

    def accumulate(
        self,
        pred_centers_vox,   # [K, 3] (x, y, z) in voxels — numpy or torch
        pred_scores,         # [K]    confidence scores in [0, 1]
        gt_points,           # [N, 5] (x, y, z, class_id, sigma_vox); padding class_id=-100
        voxel_size: float,   # Å per voxel — varies per tomogram in BYU
    ):
        """Store per-tomogram raw detections for later threshold sweep.

        Padding rows (class_id == -100) and no-motor rows are handled automatically.
        """
        if hasattr(pred_centers_vox, 'cpu'):
            pred_centers_vox = pred_centers_vox.cpu().numpy()
        if hasattr(pred_scores, 'cpu'):
            pred_scores = pred_scores.cpu().numpy()
        if hasattr(gt_points, 'cpu'):
            gt_points = gt_points.cpu().numpy()

        valid     = gt_points[:, 3] >= 0
        gt_points = gt_points[valid]

        # distance threshold in voxels (varies per tomogram because voxel_size varies)
        threshold_vox = self.min_radius / float(voxel_size)

        self._data.append({
            'pred_centers_vox': pred_centers_vox.astype(np.float32),   # [K, 3]
            'pred_scores':      pred_scores.astype(np.float32),         # [K]
            'gt_centers_vox':   gt_points[:, :3].astype(np.float32),   # [N, 3]; N=0 if no motor
            'threshold_vox':    threshold_vox,
        })

    def _fbeta_at_threshold(self, threshold: float):
        """Compute F-beta across all tomograms at a given confidence threshold.

        Uses compute_metrics per tomogram (KDTree radius matching in voxel space).
        Distance threshold varies per tomogram based on its voxel_size.
        """
        total_tp = total_fp = total_fn = 0

        for d in self._data:
            keep         = d['pred_scores'] >= threshold
            pred_centers = d['pred_centers_vox'][keep]           # [K', 3]
            gt_centers   = d['gt_centers_vox']                   # [N, 3]
            ref_radius   = d['threshold_vox']

            tp, fp, fn = compute_metrics(gt_centers, ref_radius, pred_centers)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        beta = self.beta
        prec  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec   = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        fbeta = ((1 + beta**2) * prec * rec / (beta**2 * prec + rec)
                 if (prec + rec) > 0 else 0.0)
        return fbeta, {'motor': fbeta}

    def threshold_sweep(self, n_thresholds: int = 20):
        """Sweep confidence threshold, return best F-beta and per-class breakdown.

        Returns:
            best_fbeta     : float
            best_threshold : float
            best_per_class : dict {'motor': fbeta}
        """
        thresholds = np.linspace(0.05, 1.0, n_thresholds) ** 2
        best_fbeta, best_thr, best_per_class = -1.0, float(thresholds[0]), {}
        for thr in thresholds:
            fbeta, per_class = self._fbeta_at_threshold(float(thr))
            if fbeta > best_fbeta:
                best_fbeta, best_thr, best_per_class = fbeta, float(thr), per_class
        return best_fbeta, best_thr, best_per_class

    def reset(self):
        self._data = []


# ---------------------------------------------------------------------------
# Factory  (mirrors segmentation_3d/metrics.py get_metric pattern)
# ---------------------------------------------------------------------------

def get_metric(dataset_name: str):
    if dataset_name == 'czi':
        return CZIDetectionMetrics()
    elif dataset_name == 'byu':
        return BYUDetectionMetrics()
    else:
        raise ValueError(f"Unknown detection dataset: '{dataset_name}'")
