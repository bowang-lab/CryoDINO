# Author: Ahmadreza Attarpour (attarpour1993@gmail.com)
#
# Baseline segmentation training script for CryoDINO evaluation.
# Trains MONAI UNet or UNETR from scratch (random initialization) on cryo-ET datasets
# for comparison against the pretrained CryoDINO fine-tuning pipeline.
#
# Usage:
#   Must be run from the 3DINO/ directory with PYTHONPATH=. so that the
#   dinov2 package is resolvable (matches how segmentation3d.py is launched):
#
#     cd /path/to/CryoDINO/3DINO
#     PYTHONPATH=. python ../baselines/train.py <args>
#
#   Uses standard PyTorch DataLoader (no distributed/DDP required), so plain
#   python is sufficient — no torchrun needed. The training loop cycles over
#   the dataset infinitely until max_iter (epochs * epoch_length) is reached.
#   Alternative: call torch.distributed.init_process_group(backend="nccl") at
#   startup and use torchrun to enable multi-GPU DDP training.
#
#   Full example:
#   PYTHONPATH=. python ../baselines/train.py \
#       --model_name unet \                         # or "unetr"
#       --dataset_name Dataset001_CZII_10001_patches512 \
#       --base_data_dir /cluster/projects/bwanggroup/reza/projects/cryoet/experiments \
#       --output_dir /cluster/projects/bwanggroup/reza/projects/cryoet/experiments/baselines/unet_Dataset001 \
#       --cache_dir /cluster/projects/bwanggroup/reza/projects/cryoet/experiments/cache_dir_downstream/baseline_Dataset001 \
#       --image_size 112 \
#       --epochs 100 \
#       --epoch_length 300 \
#       --eval_iters 600 \
#       --warmup_iters 3000 \
#       --learning_rate 1e-4 \
#       --batch_size 2 \
#       --num_workers 4 \
#       --dataset_percent 100
#
# Supported models:
#   unet  -- MONAI 3D UNet, channels=(16,32,64,128,256), 2 residual units per block
#   unetr -- MONAI UNETR (ViT-Base backbone, feature_size=16, hidden_size=768)
#
# Output:
#   best_model.pth  -- checkpoint with best validation Dice
#   results.json    -- training loss, val Dice per eval step, and final test Dice

import json
import os
import torch
from monai.utils import set_determinism
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceCELoss
from monai.optimizers import WarmupCosineSchedule
from monai.data.utils import list_data_collate

from dinov2.eval.segmentation_3d.augmentations import make_transforms
from dinov2.eval.segmentation_3d.metrics import get_metric
from dinov2.data.loaders import make_segmentation_dataset_3d
from torch.utils.data import DataLoader
from dinov2.eval.segmentation3d import train_iter, val_iter, clear_cuda_memory

set_determinism(seed=0)


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train baseline models for cryoet segmentation")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="unet", choices=["unet", "unetr"])
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--resize_scale", type=float, default=1.0)
    parser.add_argument("--dataset_percent", type=int, default=100)
    parser.add_argument("--base_data_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epoch_length", type=int, default=300)
    parser.add_argument("--warmup_iters", type=int, default=3000)
    parser.add_argument("--eval_iters", type=int, default=600)
    return parser.parse_args()


def get_model(model_name, num_classes, image_size):
    if model_name == "unet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
    elif model_name == "unetr":
        from monai.networks.nets import UNETR
        model = UNETR(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            img_size=(image_size, image_size, image_size),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            dropout_rate=0.2,
            norm_name="batch",
            res_block=True,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model


def print_run_info(args, num_classes):
    max_iter = args.epochs * args.epoch_length
    print("=" * 60)
    print("          CryoDINO Baseline Training")
    print("=" * 60)
    print(f"  Model            : {args.model_name.upper()}")
    print(f"  Dataset          : {args.dataset_name}")
    print(f"  Data %           : {args.dataset_percent}%")
    print(f"  Base data dir    : {args.base_data_dir}")
    print(f"  Output dir       : {args.output_dir}")
    print(f"  Cache dir        : {args.cache_dir}")
    print("-" * 60)
    print(f"  Num classes      : {num_classes}")
    print(f"  Image size       : {args.image_size}^3")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Num workers      : {args.num_workers}")
    print("-" * 60)
    print(f"  Learning rate    : {args.learning_rate}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Epoch length     : {args.epoch_length}")
    print(f"  Max iterations   : {max_iter}")
    print(f"  Warmup iters     : {args.warmup_iters}")
    print(f"  Eval every       : {args.eval_iters} iters")
    print("=" * 60)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

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
    print_run_info(args, num_classes)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        persistent_workers=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        persistent_workers=False,
    )

    seg_model = get_model(args.model_name, num_classes, args.image_size)
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, seg_model.parameters()), lr=args.learning_rate)
    max_iter = args.epochs * args.epoch_length
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_iters, t_total=max_iter)
    dice_metric = get_metric(args.dataset_name)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    scaler = torch.cuda.amp.GradScaler()

    seg_model.cuda()
    loss_fn.cuda()

    best_val_dice = -1
    train_loss_sum = 0
    iters_list = []
    train_loss_list = []
    val_dice_list = []
    val_per_cls_dice_list = []

    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    seg_model.train()
    for it, train_data in enumerate(infinite_loader(train_loader)):

        train_loss = train_iter(
            model=seg_model,
            batch=train_data,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_fn,
            scaler=scaler
        )
        train_loss_sum += train_loss

        if it % 100 == 0:
            print(f"[Iter {it}], Train loss: {train_loss}", flush=True)

        if it % args.eval_iters == 0:
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

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                print(f"Saving best model with val dice: {best_val_dice} on iter: {it}")
                torch.save(seg_model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

            seg_model.train()

        if it >= max_iter:
            break

    # test with best checkpoint
    seg_model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
    seg_model.eval()

    total_test_dice = 0
    total_per_cls_test_dice = [0 for _ in range(num_classes)]
    test_steps = 0
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

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as fp:
        json.dump({
            'iters_list': iters_list,
            'train_loss_list': train_loss_list,
            'val_dice_list': val_dice_list,
            'val_per_cls_dice_list': val_per_cls_dice_list,
            'test_dice': avg_test_dice,
            'test_per_cls_dice': avg_per_cls_test_dice,
        }, fp)


if __name__ == "__main__":
    args = get_args()
    main(args)
