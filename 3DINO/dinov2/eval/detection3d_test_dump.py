# Test-set re-evaluation of a finished detection3d.py run, saving per-tomogram predictions.
#
# Builds the same model as detection3d.py, loads <run-dir>/best_model.pth, runs the same test loop
# (sliding window, NMS decode) and writes <output-dir>/test_predictions.pkl:
#   {'images': [...], 'data': CZIDetectionMetrics._data (per tomogram: pred centers/labels/scores, GT),
#    'val_thresholds': results.json best_val_thresholds, 'test_f4': recomputed, 'test_f4_results_json': ...}
# Used for bootstrap confidence intervals. Does not touch training code or the run's results.json.

import json
import os
import pickle
from functools import partial

import numpy as np
import torch

from dinov2.data import SamplerType, make_data_loader
from dinov2.data.loaders import make_detection_dataset_3d
from dinov2.eval.detection3d import (add_seg_args, apply_lora, clear_cuda_memory, detection_collate_fn,
                                     detection_val_iter, parse_lora_targets)
from dinov2.eval.detection_3d.augmentations import make_transforms
from dinov2.eval.detection_3d.detection_heads import ResNetFPNHead, build_vit_detection_head
from dinov2.eval.detection_3d.metrics import get_metric
from dinov2.eval.setup import get_args_parser, get_autocast_dtype, setup_and_build_model_3d
from dinov2.utils.config import setup_3d


def main(args):
    with open(os.path.join(args.run_dir, "results.json")) as f:
        results = json.load(f)
    val_thr = results["best_val_thresholds"]

    if args.segmentation_head == "ResNetFPN":
        torch.backends.cudnn.benchmark = True
        autocast_dtype = get_autocast_dtype(setup_3d(args))
        feature_model = None
    else:
        feature_model, autocast_dtype = setup_and_build_model_3d(args)

    _, val_transforms = make_transforms(crop_size=args.image_size)
    _, _, test_ds, input_channels, num_classes = make_detection_dataset_3d(
        args.dataset_name, args.dataset_percent, args.base_data_dir, val_transforms, val_transforms,
        args.cache_dir, args.batch_size)
    if args.test_images:  # subset of test tomograms by basename (e.g. for qualitative figures)
        keep = set(args.test_images.split(","))
        test_ds.data = [d for d in test_ds.data if os.path.basename(d["image"]).replace(".nii.gz", "") in keep]
        assert test_ds.data, f"no test image matches {keep}"
    test_loader = make_data_loader(dataset=test_ds, batch_size=1, num_workers=args.num_workers, shuffle=False,
                                   seed=0, sampler_type=SamplerType.DISTRIBUTED, drop_last=False,
                                   persistent_workers=False, collate_fn=detection_collate_fn)

    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    if args.segmentation_head == "ViTAdapterUNETR":
        seg_model = build_vit_detection_head(feature_model, input_channels, args.image_size, num_classes,
                                             autocast_ctx, fusion=args.decoder_fusion,
                                             encoder_layers=args.encoder_layers)
        if args.lora_rank > 0:
            alpha = float(args.lora_rank) if args.lora_alpha is None else args.lora_alpha
            apply_lora(seg_model.feature_model.vit_model, args.lora_rank, alpha, parse_lora_targets(args.lora_targets))
    elif args.segmentation_head == "ResNetFPN":
        seg_model = ResNetFPNHead(input_channels, args.image_size, num_classes, autocast_ctx)
    else:
        raise ValueError(f"Unsupported head: {args.segmentation_head}")

    seg_model.load_state_dict(torch.load(os.path.join(args.run_dir, "best_model.pth"), map_location="cpu"))
    seg_model.cuda().eval()

    test_metric = get_metric(args.dataset_name)
    crops = []
    with torch.no_grad():
        for test_data in test_loader:
            off = None
            if args.patch_size_xyz:
                # Patch-only inference: the model sees only a sub-volume (volume axes = x, y, z).
                # Center: x, y = volume center; z = median GT z. Predictions/GT shifted back to
                # full-tomogram coordinates afterwards, so they overlay like full-run predictions.
                img = test_data["image"]                                   # [1, 1, X, Y, Z]
                pts = test_data["points"]                                  # [1, N, 5] (x, y, z, cls, sigma)
                shape = img.shape[2:]
                size = [int(v) for v in args.patch_size_xyz.split("x")]
                gt = pts[0][pts[0][:, 3] >= 0]
                center = [shape[0] // 2, shape[1] // 2, int(round(float(gt[:, 2].median())))]
                lo = [int(min(max(c - sz // 2, 0), dim - sz)) for c, sz, dim in zip(center, size, shape)]
                off = np.array(lo, dtype=np.float32)
                test_data = dict(test_data)
                test_data["image"] = img[:, :, lo[0]:lo[0] + size[0], lo[1]:lo[1] + size[1], lo[2]:lo[2] + size[2]]
                pts = pts.clone()
                valid = pts[..., 3] >= 0
                pts[..., :3][valid] -= torch.as_tensor(off)
                test_data["points"] = pts
                crops.append({"start_xyz": lo, "size_xyz": size})
                print(f"Patch inference: start {lo}, size {size} of volume {list(shape)}", flush=True)
            detection_val_iter(model=seg_model, batch=test_data, detection_metric=test_metric,
                               patch_size=args.image_size, stride=2, overlap=0.75, dataset_name=args.dataset_name)
            if off is not None:
                vs = float(test_data["voxel_size"][0]) if "voxel_size" in test_data else 10.0
                d = test_metric._data[-1]
                d["pred_centers_ang"] = d["pred_centers_ang"] + off * vs
                d["gt_centers_ang"] = d["gt_centers_ang"] + off * vs
            clear_cuda_memory()

    test_f4, test_per_cls = test_metric.f4_at_thresholds(val_thr)
    if args.test_images:
        print(f"Subset of {len(test_ds.data)} test tomograms: F4 {test_f4:.4f} (not comparable to full test)", flush=True)
    print(f"Recomputed test F4 @ val thresholds: {test_f4:.4f} (results.json: {results['test_f4']:.4f})", flush=True)

    out = {
        "images": [d["image"] for d in test_ds.data],
        "data": test_metric._data,
        "val_thresholds": val_thr,
        "test_f4": float(test_f4),
        "test_per_cls_f4": {k: float(v) for k, v in test_per_cls.items()},
        "test_f4_results_json": results["test_f4"],
        "patch_crops": crops,   # non-empty = patch-only inference (gt_centers_ang then only GT inside the patch)
    }
    path = os.path.join(args.output_dir, "test_predictions.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    parser = add_seg_args(get_args_parser(add_help=True))
    parser.add_argument("--run-dir", required=True, help="finished detection3d.py output dir with best_model.pth")
    parser.add_argument("--test-images", default="", help="comma-separated test basenames (no .nii.gz) to run only")
    parser.add_argument("--patch-size-xyz", default="", help="e.g. 256x256x128: run the model on this sub-volume only")
    main(parser.parse_args())
