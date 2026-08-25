#!/usr/bin/env python
"""Reusable DeePiCt 3D-U-Net trainer with a FIXED schedule: EPOCHS x ITERS_PER_EPOCH.

DeePiCt's stock training runs one pass over all partition boxes per epoch, so epoch length
varies with dataset size. This wrapper re-wraps the training DataLoader with a
replacement RandomSampler so every epoch runs EXACTLY `iterations_per_epoch` iterations,
for `epochs` epochs — a consistent training budget across any cryo-ET dataset.

It reuses DeePiCt internals (model, loss, data loaders, train/validate routines) unchanged.
Reads schedule from the config: `training.unet_hyperparameters.epochs` (default 100) and
`training.iterations_per_epoch` (default 300).

Usage:
  python deepict_train.py --config_file config.yaml --pythonpath <DeePiCt>/3d_cnn/src [--fold None]
"""
import argparse
import ast
import os
import shutil
import sys

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", required=True)
    ap.add_argument("--pythonpath", required=True, help="path to DeePiCt/3d_cnn/src")
    ap.add_argument("--fold", default="None")
    args = ap.parse_args()
    sys.path.append(args.pythonpath)

    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, RandomSampler
    from monai.losses.dice import GeneralizedDiceLoss

    from constants.config import Config, get_model_name, record_model
    from networks.io import get_device, to_device
    from networks.loss import DiceCoefficientLoss
    from networks.routines import train, validate
    from networks.unet import UNet3D
    from networks.utils import (generate_data_loaders_data_augmentation,
                                get_training_testing_lists, save_unet_model)
    from networks.visualizers import TensorBoard_multiclass

    config = Config(args.config_file)
    raw = yaml.safe_load(open(args.config_file))
    iters_per_epoch = int(raw["training"].get("iterations_per_epoch", 300))
    epochs = config.epochs                     # from unet_hyperparameters.epochs (set 100)
    fold = ast.literal_eval(args.fold)
    device = get_device()

    model_path, model_name = get_model_name(config, fold)
    best_model_path = model_path[:-4] + "_best.pth"
    last_model_path = model_path[:-4] + "_last.pth"
    os.makedirs(os.path.join(config.output_dir, "models"), exist_ok=True)
    log_path = os.path.join(config.output_dir, "logging", model_name)
    os.makedirs(log_path, exist_ok=True)

    assert config.loss in {"Dice", "GeneralizedDice"}, "loss must be Dice or GeneralizedDice"
    loss = GeneralizedDiceLoss() if config.loss == "GeneralizedDice" else DiceCoefficientLoss()
    loss = loss.to(device)
    metric = loss

    tomo_training_list, tomo_testing_list = get_training_testing_lists(config=config, fold=fold)
    model_descriptor = record_model(config=config, training_tomos=tomo_training_list,
                                    testing_tomos=tomo_testing_list, fold=fold)

    net_conf = {'final_activation': nn.Sigmoid(), 'depth': config.depth,
                'initial_features': config.initial_features,
                "out_channels": len(config.semantic_classes), "BN": config.batch_norm,
                "encoder_dropout": config.encoder_dropout, "decoder_dropout": config.decoder_dropout}
    net = UNet3D(**net_conf)
    net = to_device(net=net, gpu=None)   # gpu=None keeps SLURM's CUDA_VISIBLE_DEVICES (MIG-safe)
    optimizer = optim.Adam(net.parameters())

    # DeePiCt builds TensorDataset loaders (all boxes in RAM); re-wrap the train loader with a
    # replacement sampler so each epoch runs exactly `iters_per_epoch` iterations.
    train_loader, val_loader = generate_data_loaders_data_augmentation(
        config=config, tomo_training_list=tomo_training_list, fold=fold)
    dataset = train_loader.dataset
    sampler = RandomSampler(dataset, replacement=True,
                            num_samples=config.batch_size * iters_per_epoch)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
    assert len(train_loader) == iters_per_epoch, \
        f"expected {iters_per_epoch} iters/epoch, got {len(train_loader)}"
    print(f"SCHEDULE: {epochs} epochs x {len(train_loader)} iterations/epoch "
          f"(batch_size={config.batch_size}, boxes_in_pool={len(dataset)})", flush=True)

    logger = TensorBoard_multiclass(log_dir=log_path, log_image_interval=1)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    best_val, best_epoch = np.inf, -1
    for epoch in range(epochs):
        train(model=net, loader=train_loader, optimizer=optimizer, loss_function=loss,
              epoch=epoch, device=device, log_interval=50, tb_logger=logger,
              log_image=False, lr_scheduler=lr_scheduler)
        step = (epoch + 1) * len(dataset)
        val = validate(model=net, loader=val_loader, loss_function=loss, metric=metric,
                       device=device, step=step, tb_logger=logger, log_image_interval=None)
        if val <= best_val:
            best_val, best_epoch = val, epoch
            save_unet_model(path_to_model=model_path, epoch=epoch, net=net, optimizer=optimizer,
                            loss=val, model_descriptor=model_descriptor)
        save_unet_model(path_to_model=last_model_path, epoch=epoch, net=net, optimizer=optimizer,
                        loss=val, model_descriptor=model_descriptor)
        print(f"[epoch {epoch:3d}/{epochs}] val_loss={val:.4f} val_dice~{1 - val:.4f} "
              f"best_epoch={best_epoch}", flush=True)

    shutil.copy(model_path, best_model_path)
    print(f"DONE. best_epoch={best_epoch} best_val_loss={best_val:.4f} "
          f"val_dice~{1 - best_val:.4f} -> {best_model_path}", flush=True)


if __name__ == "__main__":
    main()
