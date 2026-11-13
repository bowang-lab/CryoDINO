# Saving test-set detections (used by detection3d.py --save-test-predictions and detection3d_test_dump.py).
#
# Writes to <output_dir>:
#   test_predictions.pkl  dict, format_version 1:
#       images, voxel_sizes            per test tomogram (same order as `data`)
#       data                           metric._data per tomogram (CZI: pred_centers_ang [K,3] (x,y,z) Å,
#                                      pred_labels [K], pred_scores [K], gt_centers_ang [N,3], gt_labels [N];
#                                      BYU: pred_centers_vox, pred_scores, gt_centers_vox, threshold_vox)
#                                      — ALL decoded candidates (score >= decode min_score), not thresholded
#       class_names, class_radii_ang, class_weights
#       val_thresholds                 per-class thresholds chosen on val (apply: score >= thr)
#       test_f4, test_per_cls_f4, ...  plus any `extra` (checkpoint, decode settings, ...)
#   test_predictions.csv  one row per candidate:
#       tomogram, x_vox, y_vox, z_vox, x_ang, y_ang, z_ang, class_id, class_name, score, above_val_threshold
#       (x/y/z_ang and score at full precision, so F4 recomputed from the CSV is exact; *_vox rounded)
# Coordinates are (x, y, z) = tomogram array axes 0, 1, 2 (nibabel order), same as the datalist points.

import csv
import os
import pickle

import numpy as np


def save_test_predictions(output_dir, images, voxel_sizes, metric, val_thresholds, extra=None):
    names = list(metric.PARTICLE_NAMES)
    is_czi = hasattr(metric, "PARTICLE_RADII_ANG")
    out = {
        "format_version": 1,
        "images": [str(i) for i in images],
        "voxel_sizes": [float(v) for v in voxel_sizes],
        "data": metric._data,
        "class_names": names,
        "class_radii_ang": list(metric.PARTICLE_RADII_ANG) if is_czi else None,
        "class_weights": list(metric.WEIGHTS) if is_czi else None,
        "val_thresholds": val_thresholds,
    }
    out.update(extra or {})
    pkl_path = os.path.join(output_dir, "test_predictions.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(out, f)

    csv_path = os.path.join(output_dir, "test_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tomogram", "x_vox", "y_vox", "z_vox", "x_ang", "y_ang", "z_ang",
                    "class_id", "class_name", "score", "above_val_threshold"])
        for img, vs, d in zip(out["images"], out["voxel_sizes"], metric._data):
            tomo = os.path.basename(img).replace(".nii.gz", "")
            if is_czi:
                ang = np.asarray(d["pred_centers_ang"], dtype=float).reshape(-1, 3)
                vox = ang / vs
                labels = np.asarray(d["pred_labels"], dtype=int)
            else:
                vox = np.asarray(d["pred_centers_vox"], dtype=float).reshape(-1, 3)
                ang = vox * vs
                labels = np.zeros(len(vox), dtype=int)
            scores = np.asarray(d["pred_scores"], dtype=float)
            for (xv, yv, zv), (xa, ya, za), c, s in zip(vox, ang, labels, scores):
                thr = val_thresholds.get(names[c]) if isinstance(val_thresholds, dict) else val_thresholds
                w.writerow([tomo, f"{xv:.3f}", f"{yv:.3f}", f"{zv:.3f}", repr(float(xa)), repr(float(ya)), repr(float(za)),
                            int(c), names[c], repr(float(s)), int(thr is not None and s >= thr)])
    return pkl_path, csv_path
