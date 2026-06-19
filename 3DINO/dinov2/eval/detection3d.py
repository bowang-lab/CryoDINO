# Author: Tony Xu
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

# Edited by AA to add 3D detection functionality. Original file is for 3D segmentation.

from dinov2.data.loaders import make_detection_dataset_3d
from dinov2.data import SamplerType, make_data_loader
from dinov2.eval.detection_3d.detection_heads import UNETRHead, LinearDecoderHead, ViTAdapterUNETRHead
from dinov2.eval.setup import get_args_parser, setup_and_build_model_3d
from dinov2.eval.detection_3d.augmentations import make_transforms
from dinov2.eval.detection_3d.metrics import get_metric
from dinov2.eval.detection_3d.loss import object_detection_loss, decode_detections_with_nms

import gc
import heapq
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from functools import partial
# AA: from monai.inferers import sliding_window_inference  # replaced by sliding_window_accumulate
from monai.optimizers import WarmupCosineSchedule


def detection_collate_fn(batch):
    images = torch.stack([d["image"] for d in batch])        # [B, 1, D, H, W]
    N_max  = max(d["points"].shape[0] for d in batch)
    labels = torch.full((len(batch), N_max, 5), -100.0)
    for i, d in enumerate(batch):
        pts = torch.as_tensor(d["points"], dtype=torch.float32)
        labels[i, :pts.shape[0]] = pts
    result = {"image": images, "points": labels}             # labels: [B, N_max, 5]  cols=(x,y,z,class_id,sigma); padding=-100
    # AA: pass voxel_size through for val/test metric computation
    if "voxel_size" in batch[0]:
        result["voxel_size"] = torch.tensor([d["voxel_size"] for d in batch], dtype=torch.float32)
    return result

class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        super().__init__()
        assert weight_factors is None or any(x != 0 for x in weight_factors), \
            "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors) if weight_factors is not None else None
        self.loss = loss

    def forward(self, *args):
        assert all(isinstance(i, (tuple, list)) for i in args), \
            f"all args must be tuple or list, got {[type(i) for i in args]}"
        weights = self.weight_factors if self.weight_factors is not None else (1,) * len(args[0])
        return sum(weights[i] * self.loss(*inputs)
                   for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0)


def clear_cuda_memory():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"Failed to clear CUDA memory: {e}")


def add_seg_args(parser):
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name of finetuning dataset",
    )
    parser.add_argument(
        "--dataset-percent",
        type=int,
        help="Percent of finetuning dataset to use",
        default=100
    )
    parser.add_argument(
        "--base-data-dir",
        type=str,
        help="Base data directory for finetuning dataset",
    )
    parser.add_argument(
        "--segmentation-head",
        type=str,
        help="Segmentation head",
    )
    parser.add_argument(
        "--train-feature-model",
        action="store_true",
        help="Freeze feature model or not",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Total epochs",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Iterations to perform per epoch",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        help="Iterations to perform per evaluation",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="Image side length",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        help="Scale factor for resizing images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="path to cache directory for monai persistent dataset"
    )
    parser.add_argument(
        "--deep-supervision",
        action="store_true",
        help="Enable deep supervision with auxiliary decoder outputs (UNETR and ViTAdapterUNETR only)",
    )

    return parser


# AA: old segmentation train_iter:
# def train_iter(model, batch, optimizer, scheduler, loss_function, scaler, deep_supervision=False):
#     x, y = (batch["image"].cuda(), batch["label"].cuda())
#     outputs = model(x)
#
#     if deep_supervision and isinstance(outputs, (list, tuple)):
#         # Downsample labels to match each auxiliary output resolution
#         labels = [F.interpolate(y.float(), size=out.shape[2:], mode='nearest') for out in outputs]
#         loss = loss_function(list(outputs), labels)
#     else:
#         loss = loss_function(outputs, y)
#
#     optimizer.zero_grad()
#     scaler.scale(loss).backward()
#     scaler.step(optimizer)
#     scaler.update()
#     scheduler.step()
#     return loss.item()

# AA: old train_iter with hardcoded strides:
# def train_iter(model, batch, optimizer, scheduler, scaler, strides=2):
#     x      = batch["image"].cuda()
#     labels = batch["points"].cuda()
#     cls_map, off_map = model(x)
#     loss, loss_dict = object_detection_loss(cls_map, off_map, strides=strides, labels=labels)
#     optimizer.zero_grad()
#     scaler.scale(loss).backward()
#     scaler.step(optimizer)
#     scaler.update()
#     scheduler.step()
#     return loss.item(), loss_dict

def train_iter(model, batch, optimizer, scheduler, scaler, loss_fn):
    x      = batch["image"].cuda()
    labels = batch["points"].cuda()
    cls_map, off_map = model(x)
    loss, loss_dict = loss_fn(cls_map, off_map, labels=labels)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.item(), loss_dict


# AA: old make_patchwise_predictor (for monai sliding_window_inference, segmentation):
# def make_patchwise_predictor(model):
#     from monai.transforms import ScaleIntensityRangePercentiles
#     normalize = ScaleIntensityRangePercentiles(
#         lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False
#     )
#     def predictor(patch_data):
#         normalized = torch.stack([normalize(patch_data[i]) for i in range(patch_data.shape[0])])
#         return model(normalized)
#     return predictor

@torch.no_grad()
def sliding_window_accumulate(model, volume, patch_size, stride=2, overlap=0.5):
    """Tile full volume, run model on each tile, accumulate class probas and off_map.

    Ported from Kaggle AccumulatedObjectDetectionPredictionContainer (od_accumulator.py).
    Raw logits are accumulated and averaged across overlapping tiles — matching Kaggle's
    od_accumulator.py. Sigmoid is applied once at decode time in decode_detections_with_nms.

    Per-patch percentile normalization is applied inside the loop, matching the
    normalization used during training (ScaleIntensityRangePercentiles 0.5-99.5 → [-1,1]).

    Args:
        model      : detection model, forward(x) → (cls_logits [1,C,D/s,H/s,W/s], off_map [1,3,...])
        volume     : [1, 1, D, H, W] float tensor (z-score normalized)
        patch_size : int, tile side length in voxels (must be divisible by stride)
        stride     : model output stride (default 2)
        overlap    : fraction of overlap between adjacent tiles (default 0.5)

    Returns:
        scores_acc  : [C, Ds, Hs, Ws] averaged logits (not yet sigmoid)
        offsets_acc : [3, Ds, Hs, Ws] averaged offset predictions
    """
    from monai.transforms import ScaleIntensityRangePercentiles
    normalize = ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False)

    _, _, D, H, W = volume.shape
    Ds, Hs, Ws    = D // stride, H // stride, W // stride
    device        = volume.device
    step          = max(1, int(patch_size * (1 - overlap)))

    def tile_positions(dim, ps):
        """Start positions for overlapping tiles along one dimension."""
        if dim <= ps:
            return [0]
        stops = list(range(0, dim - ps + 1, step))
        if stops[-1] + ps < dim:
            stops.append(dim - ps)
        return stops

    # volume is [B, C, X, Y, Z] (nibabel XYZ axis order, matching pretraining)
    # D = X dim (axis 2), H = Y dim (axis 3), W = Z dim (axis 4, thin ~60 vox)
    x_starts = tile_positions(D, patch_size)   # D = X dimension
    y_starts = tile_positions(H, patch_size)   # H = Y dimension
    z_starts = tile_positions(W, patch_size)   # W = Z dimension (thin)

    scores_acc  = None
    offsets_acc = None
    counter     = None

    for x0 in x_starts:
        for y0 in y_starts:
            for z0 in z_starts:
                x1, y1, z1 = min(x0 + patch_size, D), min(y0 + patch_size, H), min(z0 + patch_size, W)
                patch   = volume[:, :, x0:x1, y0:y1, z0:z1]                    # [1, 1, px, py, pz]
                patch_n = torch.stack([normalize(patch[i]) for i in range(patch.shape[0])])

                with torch.cuda.amp.autocast():
                    cls_logits, off = model(patch_n)                            # [1, C, px/s, py/s, pz/s]
                # AA: old per-tile sigmoid (mean(sigmoid(x)) ≠ sigmoid(mean(x))):
                # cls_logits = cls_logits[0].float().sigmoid()
                # accumulate raw logits — matches Kaggle od_accumulator.py; sigmoid applied at decode time
                cls_logits = cls_logits[0].float()                              # [C, px/s, py/s, pz/s]  logits
                off        = off[0].float()                                     # [3, px/s, py/s, pz/s]

                if scores_acc is None:
                    C           = cls_logits.shape[0]
                    scores_acc  = torch.zeros(C, Ds, Hs, Ws, device=device)
                    offsets_acc = torch.zeros(3, Ds, Hs, Ws, device=device)
                    counter     = torch.zeros(   Ds, Hs, Ws, device=device)

                x0s, y0s, z0s = x0 // stride, y0 // stride, z0 // stride
                x1s = min(x0s + cls_logits.shape[1], Ds)
                y1s = min(y0s + cls_logits.shape[2], Hs)
                z1s = min(z0s + cls_logits.shape[3], Ws)
                cx, cy, cz = x1s - x0s, y1s - y0s, z1s - z0s

                scores_acc[:, x0s:x1s, y0s:y1s, z0s:z1s]  += cls_logits[:, :cx, :cy, :cz]
                offsets_acc[:, x0s:x1s, y0s:y1s, z0s:z1s] += off[:, :cx, :cy, :cz]
                counter[x0s:x1s, y0s:y1s, z0s:z1s]        += 1

    zero_mask    = counter.eq(0).unsqueeze(0)
    scores_acc   = (scores_acc  / counter.unsqueeze(0)).masked_fill(zero_mask, 0.0)
    offsets_acc  = (offsets_acc / counter.unsqueeze(0)).masked_fill(zero_mask, 0.0)
    return scores_acc, offsets_acc


# AA: old segmentation val_iter:
# def val_iter(model, batch, metric, image_size, batch_size, overlap=0.5):
#     x, y = (batch["image"].cuda(), batch["label"].cuda())
#     logits = sliding_window_inference(x, image_size, batch_size, make_patchwise_predictor(model), overlap=overlap)
#     # logits = sliding_window_inference(x, image_size, batch_size, model, overlap=overlap)
#     iter_metric = metric(logits, y)
#     torch.cuda.empty_cache()
#     return iter_metric

def detection_val_iter(model, batch, detection_metric, patch_size, stride=2, overlap=0.5, dataset_name='czi'):
    """Run detection inference on one full tomogram via sliding window; accumulate into detection_metric.

    After iterating the full val set, call detection_metric.threshold_sweep() to find the best
    confidence threshold and F4/F-beta score, then detection_metric.reset() before the next epoch.

    Uses a low min_score (0.05) so all candidates above noise are retained; threshold_sweep()
    picks the optimal threshold post-hoc across all val tomograms simultaneously.
    """
    x          = batch["image"].cuda()              # [1, 1, D, H, W]
    gt_points  = batch["points"][0]                 # [N_max, 5] (x,y,z,class_id,sigma)
    voxel_size = float(batch["voxel_size"][0]) if "voxel_size" in batch else 10.0

    scores, offsets = sliding_window_accumulate(
        model, x, patch_size=patch_size, stride=stride, overlap=overlap
    )                                               # [C, Ds, Hs, Ws], [3, Ds, Hs, Ws]

    # NMS radius per class in voxels, derived from metric's particle radii
    if dataset_name == 'czi':
        class_sigmas = [r / voxel_size for r in detection_metric.PARTICLE_RADII_ANG]
    elif dataset_name == 'byu':
        class_sigmas = [detection_metric.min_radius / voxel_size]
    else:
        class_sigmas = [10.0]

    pred_centers, pred_labels, pred_scores = decode_detections_with_nms(
        scores=[scores],                            # list of [C, Ds, Hs, Ws] — probas (post-sigmoid)
        offsets=[offsets],                          # list of [3, Ds, Hs, Ws]
        strides=[stride],
        min_score=0.05,                             # keep all candidates; threshold_sweep picks best
        class_sigmas=class_sigmas,
        iou_threshold=0.8,                          # permissive — matches Kaggle val; nearby particles are real
        scores_are_logits=True,                     # sliding_window_accumulate returns logits; sigmoid applied here
    )

    if dataset_name == 'czi':
        detection_metric.accumulate(pred_centers, pred_labels, pred_scores, gt_points, voxel_size)
    elif dataset_name == 'byu':
        detection_metric.accumulate(pred_centers, pred_scores, gt_points, voxel_size)

    torch.cuda.empty_cache()


def do_finetune(feature_model, autocast_dtype, args):

    # get transforms, dataset, dataloaders
    # AA: old call with extra args (make_transforms signature will expand later):
    # train_transforms, val_transforms = make_transforms(
    #     args.dataset_name, args.image_size, args.resize_scale,
    #     min_int=-1.0, train_feature_model=args.train_feature_model,
    # )
    train_transforms, val_transforms = make_transforms(crop_size=args.image_size)
    train_ds, val_ds, test_ds, input_channels, num_classes = make_detection_dataset_3d(
        args.dataset_name,
        args.dataset_percent,
        args.base_data_dir,
        train_transforms,
        val_transforms,
        args.cache_dir,
        args.batch_size
    )
    train_loader = make_data_loader(
        dataset=train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=0,
        sampler_type=SamplerType.SHARDED_INFINITE,
        drop_last=False,
        persistent_workers=True,
        collate_fn=detection_collate_fn
    )
    val_loader = make_data_loader(
        dataset=val_ds,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        seed=0,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
        collate_fn=detection_collate_fn
    )
    test_loader = make_data_loader(
        dataset=test_ds,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        seed=0,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
        collate_fn=detection_collate_fn
    )

    # get model
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    scaler = torch.cuda.amp.GradScaler()
    if args.segmentation_head == 'UNETR':
        seg_model = UNETRHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx,
                              deep_supervision=args.deep_supervision)
    elif args.segmentation_head == 'Linear':
        seg_model = LinearDecoderHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx)
    elif args.segmentation_head == 'ViTAdapterUNETR':
        seg_model = ViTAdapterUNETRHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx)
    else:
        raise ValueError(f"Unknown segmentation head: {args.segmentation_head}")

    if args.train_feature_model:
        if args.segmentation_head == 'ViTAdapterUNETR':
            seg_model.feature_model.vit_model.train()
        else:
            seg_model.feature_model.train()

    else:
        if args.segmentation_head == 'ViTAdapterUNETR':
            seg_model.feature_model.vit_model.eval()
            for param in seg_model.feature_model.vit_model.parameters():
                param.requires_grad = False
        else:
            seg_model.feature_model.eval()
            for param in seg_model.feature_model.parameters():
                param.requires_grad = False

    trainable_params = [name for name, param in seg_model.named_parameters() if param.requires_grad]
    print(f"Trainable parameters: {trainable_params}")

    # get optimizer, scheduler, loss function, metric
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, seg_model.parameters()), lr=args.learning_rate)
    max_iter = args.epochs * args.epoch_length
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=args.warmup_iters,
        t_total=max_iter
    )

    # AA: old segmentation loss_fn setup (DiceCELoss/DiceLoss, deep supervision wrapper):
    # if args.dataset_name == 'BTCV' or args.dataset_name == 'LA-SEG' or ...:
    #     loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    # elif args.dataset_name == 'BraTS':
    #     loss_fn = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    # else:
    #     raise ValueError(f"Unknown dataset name: {args.dataset_name}")
    # if args.deep_supervision and args.segmentation_head in ('UNETR', 'ViTAdapterUNETR'):
    #     num_ds_outputs = 4
    #     weights = np.array([1 / (2 ** i) for i in range(num_ds_outputs)], dtype=np.float32)
    #     weights = weights / weights.sum()
    #     weights[-1] = 0.0
    #     weights = weights.tolist()
    #     print(f"Deep supervision enabled. Weights: {weights}")
    #     loss_fn = DeepSupervisionWrapper(loss_fn, weight_factors=weights)

    # Dataset-specific detection config: strides and loss hyperparameters.
    # partial() pre-fills strides so train_iter calls loss_fn(cls_map, off_map, labels=labels)
    # without needing to know the dataset. Add dataset-specific assigner params here if needed.
    if args.dataset_name == 'czi':
        detection_strides = 2
        loss_fn = partial(object_detection_loss, strides=detection_strides)
    elif args.dataset_name == 'byu':
        detection_strides = 2
        loss_fn = partial(object_detection_loss, strides=detection_strides)
    else:
        raise ValueError(f"Unknown detection dataset: '{args.dataset_name}'")

    # AA: renamed from dice_metric; returns CZIDetectionMetrics or BYUDetectionMetrics
    detection_metric = get_metric(args.dataset_name)

    seg_model.cuda()
    # AA: loss_fn.cuda()  # no separate loss_fn for detection; object_detection_loss used directly in train_iter

    # AA: old segmentation training state:
    # best_val_dice = -1
    # val_dice_list = []
    # val_per_cls_dice_list = []
    best_val_f4    = -1.0
    top5_heap      = []          # min-heap (f4, iter, ckpt_path) — keeps top-5 checkpoints
    train_loss_sum = 0
    iters_list          = []
    train_loss_list     = []
    val_f4_list         = []
    val_per_cls_f4_list = []

    for it, train_data in enumerate(train_loader):

        # train for one iteration
        # AA: segmentation train_iter call:
        # train_loss = train_iter(
        #     model=seg_model,
        #     batch=train_data,
        #     optimizer=optimizer,
        #     scheduler=scheduler,
        #     loss_function=loss_fn,
        #     scaler=scaler,
        #     deep_supervision=args.deep_supervision
        # )
        train_loss, train_loss_dict = train_iter(
            model=seg_model,
            batch=train_data,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loss_fn=loss_fn,
        )
        train_loss_sum += train_loss

        if it % 100 == 0:
            print(f"[Iter {it}], Train loss: {train_loss}", flush=True)

        if it > 0 and it % args.eval_iters == 0:
            # AA: old segmentation val block:
            # total_val_dice = 0
            # total_per_cls_val_dice = [0 for _ in range(num_classes)]
            # val_steps = 0
            # seg_model.eval()
            # with torch.no_grad():
            #     for val_data in val_loader:
            #         val_dice, val_per_cls_dice = val_iter(
            #             model=seg_model, batch=val_data,
            #             image_size=(args.image_size,) * 3, batch_size=args.batch_size,
            #             metric=dice_metric, overlap=0.
            #         )
            #         total_val_dice += val_dice
            #         for i in range(num_classes):
            #             total_per_cls_val_dice[i] += val_per_cls_dice[i]
            #         val_steps += 1
            #         clear_cuda_memory()
            # avg_val_dice = total_val_dice / val_steps
            # avg_per_cls_val_dice = [total_per_cls_val_dice[i] / val_steps for i in range(num_classes)]
            # avg_train_loss = train_loss_sum / args.eval_iters
            # train_loss_list.append(avg_train_loss)
            # val_dice_list.append(avg_val_dice)
            # val_per_cls_dice_list.append(avg_per_cls_val_dice)
            # iters_list.append(it)
            # train_loss_sum = 0
            # print(f"[Iter {it}], Train loss: {avg_train_loss}, Val dice: {avg_val_dice}")
            # print(f"Val per class dice: {avg_per_cls_val_dice}")
            # if avg_val_dice > best_val_dice:
            #     best_val_dice = avg_val_dice
            #     print(f"Saving best model with val dice: {best_val_dice} on iter: {it}")
            #     torch.save(seg_model.state_dict(), args.output_dir + "/best_model.pth")

            seg_model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    detection_val_iter(
                        model=seg_model,
                        batch=val_data,
                        detection_metric=detection_metric,
                        patch_size=args.image_size,
                        stride=detection_strides,
                        overlap=0.5,
                        dataset_name=args.dataset_name,
                    )
                    clear_cuda_memory()

            best_f4, best_thr, best_per_cls = detection_metric.threshold_sweep()
            detection_metric.reset()

            avg_train_loss = train_loss_sum / args.eval_iters
            train_loss_list.append(avg_train_loss)
            val_f4_list.append(best_f4)
            val_per_cls_f4_list.append({k: float(v) for k, v in best_per_cls.items()})
            iters_list.append(it)
            train_loss_sum = 0

            print(f"[Iter {it}] Train loss: {avg_train_loss:.4f}, Val F4: {best_f4:.4f}", flush=True)
            print(f"Val per-class F4: {best_per_cls}", flush=True)
            print(f"Val per-class thresholds: {best_thr}", flush=True)

            # top-5 checkpoint saving (min-heap keeps 5 highest F4 checkpoints)
            ckpt_path = os.path.join(args.output_dir, f"model_iter{it:07d}_f4{best_f4:.4f}.pth")
            torch.save(seg_model.state_dict(), ckpt_path)
            heapq.heappush(top5_heap, (best_f4, it, ckpt_path))
            if len(top5_heap) > 5:
                _, _, old_path = heapq.heappop(top5_heap)
                if os.path.isfile(old_path):
                    os.remove(old_path)

            if best_f4 > best_val_f4:
                best_val_f4 = best_f4
                print(f"New best Val F4: {best_val_f4:.4f} at iter {it}", flush=True)
                torch.save(seg_model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

            # set back to train mode
            seg_model.train()
            if not args.train_feature_model:
                if args.segmentation_head == 'ViTAdapterUNETR':
                    seg_model.feature_model.vit_model.eval()
                else:
                    seg_model.feature_model.eval()

        if it >= max_iter:
            break

    # AA: old segmentation test block:
    # seg_model.load_state_dict(torch.load(args.output_dir + "/best_model.pth"))
    # seg_model.eval()
    # total_test_dice = 0
    # total_per_cls_test_dice = [0 for _ in range(num_classes)]
    # test_steps = 0
    # with torch.no_grad():
    #     for test_data in test_loader:
    #         test_dice, test_per_cls_dice = val_iter(
    #             model=seg_model, batch=test_data,
    #             image_size=(args.image_size,) * 3, batch_size=args.batch_size,
    #             metric=dice_metric, overlap=0.75
    #         )
    #         total_test_dice += test_dice
    #         for i in range(num_classes):
    #             total_per_cls_test_dice[i] += test_per_cls_dice[i]
    #         test_steps += 1
    #         clear_cuda_memory()
    # avg_test_dice = total_test_dice / test_steps
    # avg_per_cls_test_dice = [total_per_cls_test_dice[i] / test_steps for i in range(num_classes)]
    # print(f"Test dice: {avg_test_dice}")
    # print(f"Test per class dice: {avg_per_cls_test_dice}")

    seg_model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
    seg_model.eval()
    test_metric = get_metric(args.dataset_name)
    with torch.no_grad():
        for test_data in test_loader:
            detection_val_iter(
                model=seg_model,
                batch=test_data,
                detection_metric=test_metric,
                patch_size=args.image_size,
                stride=detection_strides,
                overlap=0.75,
                dataset_name=args.dataset_name,
            )
            clear_cuda_memory()

    test_f4, test_thr, test_per_cls = test_metric.threshold_sweep()
    print(f"Test F4: {test_f4:.4f}", flush=True)
    print(f"Test per-class F4: {test_per_cls}", flush=True)
    print(f"Test per-class thresholds: {test_thr}", flush=True)

    # AA: old segmentation results JSON:
    # with open(f'{args.output_dir}/results.json', 'w') as fp:
    #     json.dump({
    #         'iters_list': iters_list,
    #         'train_loss_list': train_loss_list,
    #         'val_dice_list': val_dice_list,
    #         'val_per_cls_dice_list': val_per_cls_dice_list,
    #         'test_dice': avg_test_dice,
    #         'test_per_cls_dice': avg_per_cls_test_dice,
    #     }, fp)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as fp:
        json.dump({
            'iters_list':          iters_list,
            'train_loss_list':     train_loss_list,
            'val_f4_list':         val_f4_list,
            'val_per_cls_f4_list': val_per_cls_f4_list,
            'test_f4':             float(test_f4),
            'test_per_cls_f4':     {k: float(v) for k, v in test_per_cls.items()},
        }, fp)


def main(args):
    feature_model, autocast_dtype = setup_and_build_model_3d(args)
    do_finetune(feature_model, autocast_dtype, args)


if __name__ == "__main__":
    args = add_seg_args(get_args_parser(add_help=True)).parse_args()
    main(args)
