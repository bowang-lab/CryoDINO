# MONAI RetinaNet 3D detection baseline for CryoET particle picking (CZI).
#
# Pure-MONAI model (monai.apps.detection RetinaNet + ResNet-FPN, trained from scratch),
# but validated / tested with the SAME metric as detection3d.py
# (CZIDetectionMetrics: Kaggle CZII F4, radius matching in Å, class weights), so the
# numbers are directly comparable with the CryoDINO detection head.
#
# Data:
#   train     : pre-extracted 128^3 .pt patches from the honest-split datalist
#               (debug_runs_2026-09/honest-split_datalist.json, split by tomogram)
#   val/test  : FULL tomograms (imagesTr/<run>_0000.nii.gz) with global GT points from
#               point_annotations.csv — sliding-window inference over the whole volume,
#               so every particle is counted exactly once (unlike patch-level eval).
#               Tomograms are only z-scored on load; the 0.5-99.5 percentile clip to [-1, 1]
#               is applied per sliding-window tile (TileNormNet), matching the per-patch
#               normalisation of training and detection3d.py's sliding_window_accumulate.
#
# Coordinates: points are (x, y, z, class_id, sigma_vox) in voxels, nibabel XYZ order
# (= tensor dims 0, 1, 2). Boxes are MONAI "xyzxyz" in the same axis order; a particle
# of radius r (Å) becomes a cube of side 2 * r / voxel_size * box_scale.
#
# Usage (from CryoDINO/3DINO):
#   PYTHONPATH=. python dinov2/eval/detection3d_monai.py --output-dir /path/to/out

import argparse
import heapq
import json
import os
import random
import sys
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.data import Dataset, PersistentDataset
from monai.networks.nets import resnet
from monai.optimizers import WarmupCosineSchedule
from monai.transforms import Compose, MapTransform, ScaleIntensityRangePercentiles

from dinov2.eval.detection_3d.augmentations import INTENDED_SCALE_FACTORS, make_transforms
from dinov2.eval.detection_3d.metrics import get_metric

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # CryoDINO/
sys.path.insert(0, os.path.join(_REPO_ROOT, "preprocessing"))
from downstream_patch_generation import load_detection_annotations  # noqa: E402

_DATA_ROOT = "/mnt/pool/datasets/CY/cryodino"
NUM_CLASSES = 6


def get_args_parser():
    parser = argparse.ArgumentParser("MONAI RetinaNet 3D detection (CZI), CryoDINO F4 validation")
    # paths
    parser.add_argument("--datalist", type=str,
                        default=f"{_DATA_ROOT}/debug_runs_2026-09/honest-split_datalist.json",
                        help="datalist json; only its 'training' patches are used")
    parser.add_argument("--images-dir", type=str, default=f"{_DATA_ROOT}/Dataset440_CZII_10440/imagesTr",
                        help="dir with full tomograms <run>_0000.nii.gz for val/test")
    parser.add_argument("--annotations-csv", type=str,
                        default=f"{_DATA_ROOT}/Dataset440_CZII_10440/point_annotations.csv")
    parser.add_argument("--val-runs", type=str, nargs="+", default=["TS_69_2"])
    parser.add_argument("--test-runs", type=str, nargs="+", default=["TS_73_6"])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="PersistentDataset cache for training patches (default: <output-dir>/cache)")
    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epoch-length", type=int, default=300, help="iterations per epoch")
    parser.add_argument("--eval-iters", type=int, default=1500)
    parser.add_argument("--warmup-iters", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=128, help="training patch size = sliding-window roi")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    # model
    parser.add_argument("--resnet-depth", type=int, default=34, choices=[10, 18, 34, 50, 101])
    parser.add_argument("--conv1-stride", type=int, default=2)
    parser.add_argument("--returned-layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--base-anchor-shapes", type=str, default="[[8,8,8],[12,12,12],[16,16,16]]",
                        help="JSON list of anchor shapes (voxels) at the finest FPN level")
    parser.add_argument("--box-scale", type=float, default=1.0, help="box side = 2 * radius_vox * box_scale")
    # inference
    parser.add_argument("--score-thresh", type=float, default=0.02)
    parser.add_argument("--nms-thresh", type=float, default=0.22)
    parser.add_argument("--topk-candidates-per-level", type=int, default=1000)
    parser.add_argument("--detections-per-img", type=int, default=2000)
    parser.add_argument("--sw-batch-size", type=int, default=2)
    parser.add_argument("--sw-overlap", type=float, default=0.5)
    parser.add_argument("--test-sw-overlap", type=float, default=0.75)
    # debug
    parser.add_argument("--sanity-gt-as-pred", action="store_true",
                        help="feed val GT points as predictions to the metric (expects F4=1.0), then exit")
    return parser


# ---------------------------------------------------------------------------
# points <-> boxes
# ---------------------------------------------------------------------------

def points_to_boxes(points, spatial_size, box_scale=1.0):
    """[N,5] (x,y,z,cls,sigma_vox) -> boxes [N,6] xyzxyz (clipped to image), labels [N]."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 5)
    pts = pts[pts[:, 3] >= 0]
    centers = pts[:, :3]
    half = (pts[:, 4:5] * box_scale).repeat(3, axis=1)
    lo = np.clip(centers - half, 0, None)
    hi = np.minimum(centers + half, np.asarray(spatial_size, dtype=np.float32))
    boxes = np.concatenate([lo, hi], axis=1)
    return torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(pts[:, 3], dtype=torch.long)


def boxes_to_centers(boxes):
    """[K,6] xyzxyz -> [K,3] box centres (x,y,z)."""
    return (boxes[:, :3] + boxes[:, 3:]) / 2


class PointsToBoxesd(MapTransform):
    """Append MONAI detection targets 'box' / 'label' from the (augmented) 'points' array."""

    def __init__(self, box_scale=1.0):
        super().__init__(keys=["points"])
        self.box_scale = box_scale

    def __call__(self, data):
        d = dict(data)
        d["box"], d["label"] = points_to_boxes(d["points"], d["image"].shape[1:], self.box_scale)
        return d


def detection_list_collate(batch):
    return batch  # RetinaNetDetector takes lists of images / target dicts


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def make_datasets(args):
    # 0.9-1.1x intensity scale keeps training inputs on the [-1, 1] scale of the TileNormNet tiles
    # (the pretrain-matched default is ~2x); no pretrained backbone to stay consistent with here.
    train_tf, val_tf = make_transforms(crop_size=args.image_size, scale_factors=INTENDED_SCALE_FACTORS)
    train_tf = Compose([train_tf, PointsToBoxesd(args.box_scale)])

    with open(args.datalist) as f:
        train_list = json.load(f)["training"]

    sigmas_ang = dict(zip(get_metric("czi").PARTICLE_NAMES, get_metric("czi").PARTICLE_RADII_ANG))
    annotations, class_map = load_detection_annotations(args.annotations_csv, sigmas_ang, default_sigma_ang=100.0)
    assert list(class_map) == list(get_metric("czi").PARTICLE_NAMES), \
        f"CSV class order {list(class_map)} != metric class order"

    def full_tomo_list(runs):
        return [{"image": os.path.join(args.images_dir, f"{r}_0000.nii.gz"),
                 "points": annotations[r]["points"],
                 "voxel_size": annotations[r]["voxel_size"],
                 "run": r} for r in runs]

    cache_dir = args.cache_dir or os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    train_ds = PersistentDataset(train_list, transform=train_tf, cache_dir=cache_dir)
    val_ds = Dataset(full_tomo_list(args.val_runs), transform=val_tf)
    test_ds = Dataset(full_tomo_list(args.test_runs), transform=val_tf)
    print(f"# train patches: {len(train_ds)}  val tomograms: {args.val_runs}  test tomograms: {args.test_runs}")
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def build_detector(args, sw_overlap):
    base_anchor_shapes = json.loads(args.base_anchor_shapes)
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2 ** l for l in range(len(args.returned_layers) + 1)],
        base_anchor_shapes=base_anchor_shapes,
    )
    backbone = getattr(resnet, f"resnet{args.resnet_depth}")(
        spatial_dims=3, n_input_channels=1, conv1_t_stride=args.conv1_stride, pretrained=False,
    )
    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone, spatial_dims=3, pretrained_backbone=False,
        returned_layers=args.returned_layers, trainable_backbone_layers=None,
    )
    size_divisible = args.conv1_stride * 2 * 2 ** max(args.returned_layers)
    net = RetinaNet(
        spatial_dims=3,
        num_classes=NUM_CLASSES,
        num_anchors=anchor_generator.num_anchors_per_location()[0],
        feature_extractor=feature_extractor,
        size_divisible=size_divisible,
    )
    detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator)
    detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
    detector.set_hard_negative_sampler(batch_size_per_image=64, positive_fraction=0.3, pool_size=20, min_neg=16)
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_box_selector_parameters(
        score_thresh=args.score_thresh,
        topk_candidates_per_level=args.topk_candidates_per_level,
        nms_thresh=args.nms_thresh,
        detections_per_img=args.detections_per_img,
    )
    set_inferer(detector, args, sw_overlap)
    return detector


def set_inferer(detector, args, overlap):
    # Head outputs for a whole 630x630x184 tomogram are stitched on CPU to bound GPU memory.
    detector.set_sliding_window_inferer(
        roi_size=[args.image_size] * 3,
        sw_batch_size=args.sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="constant",
        sw_device="cuda",
        device="cpu",
    )


# ---------------------------------------------------------------------------
# validation with CryoDINO metric
# ---------------------------------------------------------------------------

class TileNormNet(nn.Module):
    """Percentile-normalise each sliding-window tile before the network.

    Same transform as training (per 128^3 patch) and detection3d.py's sliding_window_accumulate.
    Only used at inference, swapped in for detector.network, so checkpoint keys are unchanged.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.normalize = ScaleIntensityRangePercentiles(
            lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False
        )

    def forward(self, x):
        x = torch.stack([torch.as_tensor(self.normalize(x[i])) for i in range(x.shape[0])])
        return self.net(x)


@torch.no_grad()
def evaluate(detector, dataset, autocast_ctx, metric):
    detector.eval()
    net = detector.network
    detector.network = TileNormNet(net)
    try:
        _evaluate(detector, dataset, autocast_ctx, metric)
    finally:
        detector.network = net
    best_f4, best_thr, best_per_cls = metric.threshold_sweep()
    metric.reset()
    return best_f4, best_thr, best_per_cls


def _evaluate(detector, dataset, autocast_ctx, metric):
    for item in dataset:
        image = item["image"].as_tensor() if hasattr(item["image"], "as_tensor") else item["image"]
        with autocast_ctx():
            out = detector([image.float()], use_inferer=True)[0]
        boxes = out["box"].float().cpu()
        metric.accumulate(
            boxes_to_centers(boxes),
            out["label"].cpu(),
            out["label_scores"].float().cpu(),
            np.asarray(item["points"], dtype=np.float32).reshape(-1, 5),
            float(item["voxel_size"]),
        )
        print(f"  {item.get('run', '')}: {boxes.shape[0]} detections, "
              f"{int((item['points'][:, 3] >= 0).sum())} GT", flush=True)
        torch.cuda.empty_cache()


def sanity_gt_as_pred(dataset):
    metric = get_metric("czi")
    for item in dataset:
        pts = np.asarray(item["points"], dtype=np.float32).reshape(-1, 5)
        pts = pts[pts[:, 3] >= 0]
        boxes, labels = points_to_boxes(pts, item["image"].shape[1:])
        metric.accumulate(boxes_to_centers(boxes), labels, torch.ones(len(labels)), pts, float(item["voxel_size"]))
    best_f4, _, best_per_cls = metric.threshold_sweep()
    print(f"[sanity] GT-as-pred F4: {best_f4:.4f}  per-class: {best_per_cls}", flush=True)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def do_train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    train_ds, val_ds, test_ds = make_datasets(args)
    if args.sanity_gt_as_pred:
        sanity_gt_as_pred(val_ds)
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=detection_list_collate, drop_last=True, persistent_workers=args.num_workers > 0,
    )

    detector = build_detector(args, args.sw_overlap).to(device)
    n_params = sum(p.numel() for p in detector.network.parameters())
    print(f"RetinaNet resnet{args.resnet_depth}-FPN: {n_params / 1e6:.2f}M params", flush=True)

    optimizer = torch.optim.AdamW(detector.network.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    max_iter = args.epochs * args.epoch_length
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_iters, t_total=max_iter)
    use_amp = not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_ctx = partial(torch.autocast, device_type="cuda", dtype=torch.float16, enabled=use_amp)

    val_metric = get_metric("czi")
    iters_list, train_loss_list, val_f4_list, val_per_cls_f4_list, val_thr_list = [], [], [], [], []
    best_val_f4 = -1.0
    top5_heap = []
    train_loss_sum, train_loss_count = 0.0, 0
    window = {"loss": 0.0, "cls": 0.0, "reg": 0.0, "n": 0}

    detector.train()
    for it, batch in enumerate(infinite(train_loader), start=1):
        images = [d["image"].as_tensor().float().to(device) if hasattr(d["image"], "as_tensor")
                  else d["image"].float().to(device) for d in batch]
        targets = [{"box": d["box"].to(device), "label": d["label"].to(device)} for d in batch]

        with autocast_ctx():
            losses = detector(images, targets)
            loss = losses[detector.cls_key] + losses[detector.box_reg_key]

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(detector.network.parameters(), max_norm=12.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss_sum += loss.item()
        train_loss_count += 1
        window["loss"] += loss.item()
        window["cls"] += losses[detector.cls_key].item()
        window["reg"] += losses[detector.box_reg_key].item()
        window["n"] += 1

        if it % 100 == 0 or it == max_iter:
            n = window["n"]
            print(f"[Iter {it}] train loss (mean/{n}): {window['loss'] / n:.4f}  "
                  f"cls: {window['cls'] / n:.4f}  reg: {window['reg'] / n:.4f}  "
                  f"lr: {scheduler.get_last_lr()[0]:.3e}", flush=True)
            window = {"loss": 0.0, "cls": 0.0, "reg": 0.0, "n": 0}

        if it % args.eval_iters == 0 or it == max_iter:
            best_f4, best_thr, best_per_cls = evaluate(detector, val_ds, autocast_ctx, val_metric)

            avg_train_loss = train_loss_sum / max(train_loss_count, 1)
            train_loss_list.append(avg_train_loss)
            val_f4_list.append(float(best_f4))
            val_per_cls_f4_list.append({k: float(v) for k, v in best_per_cls.items()})
            val_thr_list.append({k: float(v) for k, v in best_thr.items()})
            iters_list.append(it)
            train_loss_sum, train_loss_count = 0.0, 0

            print(f"[Iter {it}] Train loss: {avg_train_loss:.4f}, Val F4: {best_f4:.4f}", flush=True)
            print(f"Val per-class F4: {best_per_cls}", flush=True)
            print(f"Val per-class thresholds: {best_thr}", flush=True)

            # top-5 checkpoint saving (min-heap keeps 5 highest F4 checkpoints)
            ckpt_path = os.path.join(args.output_dir, f"model_iter{it:07d}_f4{best_f4:.4f}.pth")
            torch.save(detector.network.state_dict(), ckpt_path)
            heapq.heappush(top5_heap, (best_f4, it, ckpt_path))
            if len(top5_heap) > 5:
                _, _, old_path = heapq.heappop(top5_heap)
                if os.path.isfile(old_path):
                    os.remove(old_path)

            if best_f4 > best_val_f4:
                best_val_f4 = best_f4
                print(f"New best Val F4: {best_val_f4:.4f} at iter {it}", flush=True)
                torch.save(detector.network.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

            detector.train()

        if it >= max_iter:
            break

    # test with best model
    detector.network.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
    set_inferer(detector, args, args.test_sw_overlap)
    test_f4, test_thr, test_per_cls = evaluate(detector, test_ds, autocast_ctx, get_metric("czi"))
    print(f"Test F4: {test_f4:.4f}", flush=True)
    print(f"Test per-class F4: {test_per_cls}", flush=True)
    print(f"Test per-class thresholds: {test_thr}", flush=True)

    with open(os.path.join(args.output_dir, "results.json"), "w") as fp:
        json.dump({
            "iters_list":          iters_list,
            "train_loss_list":     train_loss_list,
            "val_f4_list":         val_f4_list,
            "val_per_cls_f4_list": val_per_cls_f4_list,
            "val_per_cls_thr_list": val_thr_list,
            "test_f4":             float(test_f4),
            "test_per_cls_f4":     {k: float(v) for k, v in test_per_cls.items()},
            "test_per_cls_thr":    {k: float(v) for k, v in test_thr.items()},
            "args":                vars(args),
        }, fp, indent=2)


if __name__ == "__main__":
    do_train(get_args_parser().parse_args())
