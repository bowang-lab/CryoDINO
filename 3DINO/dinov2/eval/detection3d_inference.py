# Loads a fine-tuned detection checkpoint (e.g. best_model.pth produced by
# detection3d.py) and runs sliding-window inference over FULL tomograms. For each
# tomogram it decodes detections (NMS) and writes them to a per-run CSV. If GT
# points are present it also runs the same F-beta threshold-sweep as training and
# writes a results JSON.
#
# Two input modes:
#   (A) --raw-dataset-dir <Dataset***>   (recommended for tomo-level inference)
#         Scans <dir>/imagesTr/*.nii.gz as full tomograms. GT (+ per-run
#         voxel_size) is read from <dir>/point_annotations.csv if present
#         (or --csv). Works for the hidden test set (no CSV → predictions only).
#   (B) --split {training,validation,test}
#         Uses the patch datalist ({base-data-dir}/{dataset}_100_datalist.json).
#         NOTE: its `training` split is 128^3 .pt PATCHES, not full tomograms —
#         use mode (A) for tomogram-level train inference.
#
# All heavy lifting (model build, sliding-window accumulate, decode+NMS, metric)
# is imported from detection3d.py / the detection_3d package so inference stays
# in lock-step with training.
#
# CLASS ORDERING NOTE (important): the model is trained on ALPHABETICAL class IDs
# (load_detection_annotations sorts particle names). The prediction CSVs below are
# therefore named with the alphabetical convention, which is the model's true
# output semantics. CZIDetectionMetrics, however, lists classes in the official
# Kaggle order — see the warning printed at the end of a GT run.

import argparse
import csv
import glob
import json
import os
from functools import partial as _partial

import numpy as np
import torch

from monai.data import Dataset

from dinov2.data import SamplerType, make_data_loader
from dinov2.data.loaders import make_detection_dataset_3d
from dinov2.eval.setup import get_args_parser, setup_and_build_model_3d
from dinov2.eval.detection_3d.augmentations import make_transforms
from dinov2.eval.detection_3d.metrics import get_metric
from dinov2.eval.detection_3d.detection_heads import (
    UNETRHead,
    LinearDecoderHead,
    ViTAdapterUNETRHead,
)
from dinov2.eval.detection_3d.loss import decode_detections_with_nms

# Reuse the exact training-time helpers so inference matches training behaviour.
from dinov2.eval.detection3d import (
    add_seg_args,
    detection_collate_fn,
    sliding_window_accumulate,
    clear_cuda_memory,
)


# Model class IDs are alphabetical (see load_detection_annotations). This is the
# model's true output-channel → particle mapping for CZI.
CZI_CLASS_NAMES_ALPHA = [
    "Beta-amylase",        # 0
    "Beta-galactosidase",  # 1
    "Thyroglobulin",       # 2
    "cytosolic ribosome",  # 3
    "ferritin complex",    # 4
    "virus-like capsid",   # 5
]
BYU_CLASS_NAMES_ALPHA = ["motor"]


# ---------------------------------------------------------------------------
# CSV annotation loader (copied from preprocessing/downstream_patch_generation.py
# to avoid a cross-directory import — cwd is 3DINO/ at run time).
# ---------------------------------------------------------------------------

def _run_name_from_nii(nii_path: str) -> str:
    """'TS_5_4_0000.nii.gz' → 'TS_5_4'  (strips nnU-Net _XXXX channel suffix)."""
    base = os.path.basename(nii_path).replace(".nii.gz", "").replace(".nii", "")
    if "_" in base and len(base) >= 5 and base[-5] == "_" and base[-4:].isdigit():
        base = base[:-5]
    return base


def _load_detection_annotations(csv_path, sigmas_ang, default_sigma_ang):
    """Load point annotation CSV → ({run: {points, voxel_size}}, class_map).

    CSV format: run, particle_name, z, y, x, voxel_size  (z/y/x in voxels).
    Stored as (x_vox, y_vox, z_vox, class_id, sigma_vox); class_id is alphabetical
    over unique particle names — matching how the model was trained.
    """
    rows = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))

    particle_names = sorted({r["particle_name"].strip() for r in rows})
    class_map = {name: idx for idx, name in enumerate(particle_names)}

    annotations = {}
    for row in rows:
        run = row["run"].strip()
        particle = row["particle_name"].strip()
        z_vox, y_vox, x_vox = float(row["z"]), float(row["y"]), float(row["x"])
        vs = float(row["voxel_size"])
        class_id = float(class_map[particle])
        sigma_vox = sigmas_ang.get(particle, default_sigma_ang) / vs if vs else 0.0
        annotations.setdefault(run, {"points": [], "voxel_size": vs})
        annotations[run]["points"].append([x_vox, y_vox, z_vox, class_id, sigma_vox])

    return annotations, class_map


def add_inference_args(parser):
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Fine-tuned detection checkpoint (state_dict). "
             "Defaults to {output-dir}/best_model.pth",
    )
    # ---- mode A: raw dataset directory ----
    parser.add_argument(
        "--raw-dataset-dir", type=str, default=None,
        help="Raw Dataset*** dir for tomo-level inference. Scans <dir>/<images-subdir>/*.nii.gz. "
             "If set, this overrides the --split datalist path.",
    )
    parser.add_argument("--images-subdir", type=str, default="imagesTr",
                        help="Subdir of --raw-dataset-dir holding .nii.gz tomograms (default imagesTr)")
    parser.add_argument("--csv", type=str, default=None,
                        help="point_annotations.csv for GT. Default: <raw-dataset-dir>/point_annotations.csv if present")
    parser.add_argument("--sigmas-ang", type=str, default=None,
                        help='JSON map particle_name→radius_Å (GT sigma; not used by the CZI metric)')
    parser.add_argument("--default-sigma-ang", type=float, default=0.0,
                        help="Fallback sigma (Å) for particles not in --sigmas-ang")
    parser.add_argument("--default-voxel-size", type=float, default=10.0,
                        help="Å/voxel for tomograms with no CSV entry (e.g. hidden test). Default 10.0")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Label for output filenames in raw mode (default: basename of --raw-dataset-dir)")
    # ---- mode B: datalist split ----
    parser.add_argument("--split", type=str, default="test",
                        choices=["training", "validation", "test"],
                        help="Datalist split for mode B (default: test)")
    # ---- shared inference knobs ----
    parser.add_argument("--overlap", type=float, default=0.75,
                        help="Sliding-window tile overlap (default 0.75, matching training test eval)")
    parser.add_argument("--min-score", type=float, default=0.05,
                        help="Keep candidates above this score (default 0.05)")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.8,
                        help="Gaussian IOU above which nearby detections are suppressed (default 0.8)")
    parser.add_argument("--predictions-csv", type=str, default=None,
                        help="Output CSV path (default: {output-dir}/predictions_{tag}.csv)")
    parser.add_argument("--results-json", type=str, default=None,
                        help="Output metrics JSON path, GT only (default: {output-dir}/results_inference_{tag}.json)")
    return parser


def build_seg_model(feature_model, autocast_dtype, args, input_channels, num_classes):
    autocast_ctx = _partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    if args.segmentation_head == "UNETR":
        return UNETRHead(feature_model, input_channels, args.image_size, num_classes,
                         autocast_ctx, deep_supervision=args.deep_supervision)
    if args.segmentation_head == "Linear":
        return LinearDecoderHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx)
    if args.segmentation_head == "ViTAdapterUNETR":
        return ViTAdapterUNETRHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx)
    raise ValueError(f"Unknown segmentation head: {args.segmentation_head}")


def num_classes_for(dataset_name):
    if dataset_name == "czi":
        return 6
    if dataset_name == "byu":
        return 1
    raise ValueError(f"Unknown detection dataset: '{dataset_name}'")


def build_raw_dataset(args, val_transforms):
    """Build a full-tomogram Dataset from a raw Dataset*** dir.

    Returns (dataset, tomo_ids, label_names). label_names maps model class id →
    particle name (alphabetical class_map from the CSV, else the hardcoded
    alphabetical CZI/BYU list).
    """
    images_dir = os.path.join(args.raw_dataset_dir, args.images_subdir)
    nii_paths = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")) +
                       glob.glob(os.path.join(images_dir, "*.nii")))
    if not nii_paths:
        raise FileNotFoundError(f"No .nii/.nii.gz tomograms found under {images_dir}")

    csv_path = args.csv or os.path.join(args.raw_dataset_dir, "point_annotations.csv")
    sigmas_ang = json.loads(args.sigmas_ang) if args.sigmas_ang else {}

    annotations, class_map = {}, None
    if os.path.isfile(csv_path):
        annotations, class_map = _load_detection_annotations(
            csv_path, sigmas_ang, args.default_sigma_ang)
        print(f"Loaded GT from {csv_path}: {len(annotations)} annotated runs; class_map={class_map}")
    else:
        print(f"No CSV at {csv_path} — running predictions-only (no GT metric).")

    entries, tomo_ids = [], []
    for nii in nii_paths:
        run = _run_name_from_nii(nii)
        info = annotations.get(run)
        if info and info["points"]:
            points = info["points"]
            vs = float(info["voxel_size"])
        else:
            points = [[-100.0, -100.0, -100.0, -100.0, -100.0]]  # padding row (no GT)
            vs = float(info["voxel_size"]) if info else float(args.default_voxel_size)
        entries.append({"image": nii, "points": points, "voxel_size": vs})
        tomo_ids.append(run)

    dataset = Dataset(entries, transform=val_transforms)

    if class_map is not None:
        label_names = [name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    else:
        label_names = (CZI_CLASS_NAMES_ALPHA if args.dataset_name == "czi"
                       else BYU_CLASS_NAMES_ALPHA)
    return dataset, tomo_ids, label_names


def _tomo_ids_from_persistent(dataset, n):
    raw = getattr(dataset, "data", None)
    ids = []
    if raw is not None:
        for entry in raw:
            img = entry.get("image", "") if isinstance(entry, dict) else ""
            ids.append(_run_name_from_nii(str(img)) if img else f"tomo_{len(ids):04d}")
    while len(ids) < n:
        ids.append(f"tomo_{len(ids):04d}")
    return ids


@torch.no_grad()
def run_inference(args):
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = args.checkpoint or os.path.join(args.output_dir, "best_model.pth")
    detection_strides = 2  # czi and byu both use stride 2

    feature_model, autocast_dtype = setup_and_build_model_3d(args)
    _, val_transforms = make_transforms(crop_size=args.image_size)

    # ---- build dataset (mode A: raw dir, or mode B: datalist split) ----
    if args.raw_dataset_dir:
        tag = args.run_name or os.path.basename(os.path.normpath(args.raw_dataset_dir))
        dataset, tomo_ids, label_names = build_raw_dataset(args, val_transforms)
        num_classes = num_classes_for(args.dataset_name)
        input_channels = 1
    else:
        tag = args.split
        train_ds, val_ds, test_ds, input_channels, num_classes = make_detection_dataset_3d(
            args.dataset_name, args.dataset_percent, args.base_data_dir,
            val_transforms, val_transforms, args.cache_dir, args.batch_size)
        dataset = {"training": train_ds, "validation": val_ds, "test": test_ds}[args.split]
        tomo_ids = _tomo_ids_from_persistent(dataset, len(dataset))
        label_names = (CZI_CLASS_NAMES_ALPHA if args.dataset_name == "czi"
                       else BYU_CLASS_NAMES_ALPHA)

    predictions_csv = args.predictions_csv or os.path.join(args.output_dir, f"predictions_{tag}.csv")
    results_json = args.results_json or os.path.join(args.output_dir, f"results_inference_{tag}.json")

    # ---- model + checkpoint ----
    seg_model = build_seg_model(feature_model, autocast_dtype, args, input_channels, num_classes)
    print(f"Loading fine-tuned checkpoint: {checkpoint}", flush=True)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = seg_model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys (e.g. {missing[:3]})", flush=True)
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})", flush=True)
    seg_model.cuda().eval()

    loader = make_data_loader(
        dataset=dataset, batch_size=1, num_workers=args.num_workers, shuffle=False,
        seed=0, sampler_type=SamplerType.DISTRIBUTED, drop_last=False,
        persistent_workers=False, collate_fn=detection_collate_fn)

    metric = get_metric(args.dataset_name)
    has_any_gt = False
    n_total = len(dataset)

    fieldnames = ["tomo_id", "class_id", "particle_type",
                  "x_vox", "y_vox", "z_vox", "x_ang", "y_ang", "z_ang", "score"]
    csv_rows = []

    for idx, batch in enumerate(loader):
        x = batch["image"].cuda()
        gt_points = batch["points"][0]
        voxel_size = float(batch["voxel_size"][0]) if "voxel_size" in batch else args.default_voxel_size
        tomo_id = tomo_ids[idx] if idx < len(tomo_ids) else f"tomo_{idx:04d}"

        scores, offsets = sliding_window_accumulate(
            seg_model, x, patch_size=args.image_size, stride=detection_strides, overlap=args.overlap)

        if args.dataset_name == "czi":
            class_sigmas = [r / voxel_size for r in metric.PARTICLE_RADII_ANG]
        elif args.dataset_name == "byu":
            class_sigmas = [metric.min_radius / voxel_size]
        else:
            class_sigmas = [10.0]

        pred_centers, pred_labels, pred_scores = decode_detections_with_nms(
            scores=[scores], offsets=[offsets], strides=[detection_strides],
            min_score=args.min_score, class_sigmas=class_sigmas,
            iou_threshold=args.nms_iou_threshold, scores_are_logits=True)

        pc, pl, ps = (pred_centers.cpu().numpy(), pred_labels.cpu().numpy(), pred_scores.cpu().numpy())
        for (xv, yv, zv), lbl, sc in zip(pc, pl, ps):
            lbl = int(lbl)
            name = label_names[lbl] if lbl < len(label_names) else str(lbl)
            csv_rows.append({
                "tomo_id": tomo_id, "class_id": lbl, "particle_type": name,
                "x_vox": float(xv), "y_vox": float(yv), "z_vox": float(zv),
                "x_ang": float(xv) * voxel_size, "y_ang": float(yv) * voxel_size,
                "z_ang": float(zv) * voxel_size, "score": float(sc),
            })

        if (gt_points[:, 3] >= 0).sum().item() > 0:
            has_any_gt = True
            if args.dataset_name == "czi":
                metric.accumulate(pred_centers, pred_labels, pred_scores, gt_points, voxel_size)
            else:
                metric.accumulate(pred_centers, pred_scores, gt_points, voxel_size)

        print(f"[{idx + 1}/{n_total}] {tomo_id}: {len(ps)} detections (voxel_size={voxel_size} Å)", flush=True)
        clear_cuda_memory()

    with open(predictions_csv, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nWrote {len(csv_rows)} detections to {predictions_csv}", flush=True)

    if has_any_gt:
        best_f4, best_thr, best_per_cls = metric.threshold_sweep()
        print(f"\n[{tag}] aggregate F-beta: {best_f4:.4f}", flush=True)
        print(f"Per-class F-beta: {best_per_cls}", flush=True)
        print(f"Per-class thresholds: {best_thr}", flush=True)
        with open(results_json, "w") as fp:
            json.dump({
                "tag": tag, "checkpoint": checkpoint, "f_beta": float(best_f4),
                "per_class_f_beta": {k: float(v) for k, v in best_per_cls.items()},
                "per_class_thresholds": {k: float(v) for k, v in best_thr.items()},
                "overlap": args.overlap, "nms_iou_threshold": args.nms_iou_threshold,
            }, fp, indent=2)
        print(f"Wrote metrics to {results_json}", flush=True)
        if args.dataset_name == "czi":
            print("\n[!] NOTE: CZIDetectionMetrics class order (official Kaggle) differs from the "
                  "model's alphabetical class IDs. The aggregate matches training's reported F4 "
                  "(apples-to-apples), but per-class radii/weights are permuted vs the official "
                  "metric. The predictions CSV uses the correct alphabetical names.", flush=True)
    else:
        print("\nNo GT found — skipped metric (predictions-only run).", flush=True)


def main():
    parser = get_args_parser(add_help=True)
    parser = add_seg_args(parser)
    parser = add_inference_args(parser)
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
