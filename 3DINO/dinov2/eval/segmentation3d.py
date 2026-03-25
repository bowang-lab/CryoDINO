# Author: Tony Xu
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from dinov2.data.loaders import make_segmentation_dataset_3d
from dinov2.data import SamplerType, make_data_loader
from dinov2.eval.segmentation_3d.segmentation_heads import UNETRHead, LinearDecoderHead, ViTAdapterUNETRHead
from dinov2.eval.setup import get_args_parser, setup_and_build_model_3d
from dinov2.eval.segmentation_3d.augmentations import make_transforms
from dinov2.eval.segmentation_3d.metrics import get_metric

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from functools import partial
from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
from monai.data.utils import list_data_collate
from monai.optimizers import WarmupCosineSchedule


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


def train_iter(model, batch, optimizer, scheduler, loss_function, scaler, deep_supervision=False):
    x, y = (batch["image"].cuda(), batch["label"].cuda())
    outputs = model(x)

    if deep_supervision and isinstance(outputs, (list, tuple)):
        # Downsample labels to match each auxiliary output resolution
        labels = [F.interpolate(y.float(), size=out.shape[2:], mode='nearest') for out in outputs]
        loss = loss_function(list(outputs), labels)
    else:
        loss = loss_function(outputs, y)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.item()


def make_patchwise_predictor(model):
    """Wrap model to normalize each sliding window patch before forward pass,
    matching per-patch percentile normalization used during pretraining."""
    from monai.transforms import ScaleIntensityRangePercentiles
    normalize = ScaleIntensityRangePercentiles(
        lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False
    )

    def predictor(patch_data):
        # patch_data: (N, C, H, W, D) from sliding_window_inference
        # normalize each sample in the batch independently, matching per-crop training normalization
        normalized = torch.stack([normalize(patch_data[i]) for i in range(patch_data.shape[0])])
        return model(normalized)
    return predictor


def val_iter(model, batch, metric, image_size, batch_size, overlap=0.5):
    x, y = (batch["image"].cuda(), batch["label"].cuda())
    logits = sliding_window_inference(x, image_size, batch_size, make_patchwise_predictor(model), overlap=overlap)
    # logits = sliding_window_inference(x, image_size, batch_size, model, overlap=overlap)

    iter_metric = metric(logits, y)
    torch.cuda.empty_cache()
    return iter_metric


def do_finetune(feature_model, autocast_dtype, args):

    # get transforms, dataset, dataloaders
    train_transforms, val_transforms = make_transforms(
        args.dataset_name,
        args.image_size,
        args.resize_scale,
        min_int=-1.0
    )
    train_ds, val_ds, test_ds, input_channels, num_classes = make_segmentation_dataset_3d(
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
        collate_fn=list_data_collate
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
        collate_fn=list_data_collate
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
        collate_fn=list_data_collate
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
        seg_model = ViTAdapterUNETRHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx,
                                        deep_supervision=args.deep_supervision)
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

    if args.dataset_name == 'BTCV' or args.dataset_name == 'LA-SEG' or args.dataset_name == 'TDSC-ABUS' or 'Dataset' in args.dataset_name:
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    elif args.dataset_name == 'BraTS':
        loss_fn = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")

    if args.deep_supervision and args.segmentation_head in ('UNETR', 'ViTAdapterUNETR'):
        # 4 outputs: full res, 1/2, 1/4, 1/8
        # Normalize first, then zero the lowest resolution output
        num_ds_outputs = 4
        weights = np.array([1 / (2 ** i) for i in range(num_ds_outputs)], dtype=np.float32)
        weights = weights / weights.sum()
        weights[-1] = 0.0
        weights = weights.tolist()
        print(f"Deep supervision enabled. Weights: {weights}")
        loss_fn = DeepSupervisionWrapper(loss_fn, weight_factors=weights)

    dice_metric = get_metric(args.dataset_name)

    seg_model.cuda()
    loss_fn.cuda()

    best_val_dice = -1
    train_loss_sum = 0
    iters_list = []
    train_loss_list = []
    val_dice_list = []
    val_per_cls_dice_list = []

    for it, train_data in enumerate(train_loader):

        # train for one iteration
        train_loss = train_iter(
            model=seg_model,
            batch=train_data,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_fn,
            scaler=scaler,
            deep_supervision=args.deep_supervision
        )
        train_loss_sum += train_loss

        if it % 100 == 0:
            print(f"[Iter {it}], Train loss: {train_loss}", flush=True)

        if it % args.eval_iters == 0:
            # valdation
            total_val_dice = 0
            total_per_cls_val_dice = [0 for _ in range(num_classes)]
            val_steps = 0
            seg_model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_dice, val_per_cls_dice = val_iter(
                        model=seg_model,
                        batch=val_data,
                        image_size=(args.image_size,) * 3,
                        batch_size=args.batch_size,
                        metric=dice_metric,
                        overlap=0.
                    )

                    total_val_dice += val_dice
                    for i in range(num_classes):
                        total_per_cls_val_dice[i] += val_per_cls_dice[i]
                    val_steps += 1
                    clear_cuda_memory()

            avg_val_dice = total_val_dice / val_steps
            avg_per_cls_val_dice = [total_per_cls_val_dice[i] / val_steps for i in range(num_classes)]
            avg_train_loss = train_loss_sum / args.eval_iters

            train_loss_list.append(avg_train_loss)
            val_dice_list.append(avg_val_dice)
            val_per_cls_dice_list.append(avg_per_cls_val_dice)
            iters_list.append(it)
            train_loss_sum = 0

            print(f"[Iter {it}], Train loss: {avg_train_loss}, Val dice: {avg_val_dice}")
            print(f"Val per class dice: {avg_per_cls_val_dice}")

            # save best model
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                print(f"Saving best model with val dice: {best_val_dice} on iter: {it}")
                torch.save(seg_model.state_dict(), args.output_dir + "/best_model.pth")

            # set back to train mode
            seg_model.train()
            if not args.train_feature_model:
                if args.segmentation_head == 'ViTAdapterUNETR':
                    seg_model.feature_model.vit_model.eval()
                else:
                    seg_model.feature_model.eval()

        if it >= max_iter:
            break

    # test
    seg_model.load_state_dict(torch.load(args.output_dir + "/best_model.pth"))
    seg_model.eval()

    total_test_dice = 0
    total_per_cls_test_dice = [0 for _ in range(num_classes)]
    test_steps = 0
    seg_model.eval()
    with torch.no_grad():
        for test_data in test_loader:
            test_dice, test_per_cls_dice = val_iter(
                model=seg_model,
                batch=test_data,
                image_size=(args.image_size,) * 3,
                batch_size=args.batch_size,
                metric=dice_metric,
                overlap=0.75
            )

            total_test_dice += test_dice
            for i in range(num_classes):
                total_per_cls_test_dice[i] += test_per_cls_dice[i]
            test_steps += 1
            clear_cuda_memory()

    avg_test_dice = total_test_dice / test_steps
    avg_per_cls_test_dice = [total_per_cls_test_dice[i] / test_steps for i in range(num_classes)]

    print(f"Test dice: {avg_test_dice}")
    print(f"Test per class dice: {avg_per_cls_test_dice}")

    with open(f'{args.output_dir}/results.json', 'w') as fp:
        json.dump({
            'iters_list': iters_list,
            'train_loss_list': train_loss_list,
            'val_dice_list': val_dice_list,
            'val_per_cls_dice_list': val_per_cls_dice_list,
            'test_dice': avg_test_dice,
            'test_per_cls_dice': avg_per_cls_test_dice,
        }, fp)


def main(args):
    feature_model, autocast_dtype = setup_and_build_model_3d(args)
    do_finetune(feature_model, autocast_dtype, args)


if __name__ == "__main__":
    args = add_seg_args(get_args_parser(add_help=True)).parse_args()
    main(args)
