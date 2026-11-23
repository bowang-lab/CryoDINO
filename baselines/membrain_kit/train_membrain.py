"""Train membrain-seg (binary) on NIfTI patches, with control over iterations-per-epoch.

Thin wrapper over membrain-seg's own DataModule + U-Net so we can:
  * set a fixed number of iterations per epoch (membrain's CLI can't); with few patches a
    natural epoch is tiny, so we oversample with a replacement RandomSampler -> each epoch
    runs `--iters-per-epoch` fresh random crops (the nnUNet paradigm).
  * use bf16 + gradient clipping (fp16 + SGD momentum 0.99 tends to NaN on some data).

Expects a membrain data_dir:
    data_dir/{imagesTr,labelsTr,imagesVal,labelsVal}/*.nii.gz
with the membrain pairing rule: label `X.nii.gz`  <->  image `X_0000.nii.gz`.
Labels must be binary {0,1} (background/foreground). Value 2 is membrain's ignore label;
do NOT let whole crops be all-2 or the Dice/CE loss divides by zero -> NaN.

Use --smoke for a fast GPU sanity run that asserts train_loss is FINITE.
"""
import argparse

import pytorch_lightning as pl
from monai.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import RandomSampler

from membrain_seg.segmentation.dataloading.memseg_pl_datamodule import (
    MemBrainSegDataModule,
)
from membrain_seg.segmentation.networks.unet import SemanticSegmentationUnet
from membrain_seg.segmentation.training.optim_utils import PrintLearningRate


class OversampledDataModule(MemBrainSegDataModule):
    """membrain's DataModule, but the train loader yields `iters_per_epoch` batches of
    fresh random crops drawn with replacement from the patch pool."""

    def __init__(self, *args, iters_per_epoch=300, **kwargs):
        super().__init__(*args, **kwargs)
        self.iters_per_epoch = iters_per_epoch

    def train_dataloader(self):
        sampler = RandomSampler(
            self.train_dataset,
            replacement=True,
            num_samples=self.batch_size * self.iters_per_epoch,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
        )


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="membrain data_dir with imagesTr/labelsTr/imagesVal/labelsVal")
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--iters-per-epoch", type=int, default=300,
                    help="fixed number of training iterations (random crops) per epoch")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--project-name", default="membrain")
    ap.add_argument("--sub-name", default="binary")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--precision", default="bf16-mixed",
                    help="bf16-mixed (recommended on A100/H100). 16-mixed (fp16) can NaN with SGD.")
    ap.add_argument("--grad-clip", type=float, default=12.0)
    ap.add_argument("--no-aug", action="store_true",
                    help="disable full augmentation (faster epochs, worse generalization)")
    ap.add_argument("--no-deep-supervision", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="fast GPU sanity run: 1 epoch, few batches, asserts finite train_loss")
    return ap


def main():
    args = build_arg_parser().parse_args()

    dm = OversampledDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0 if args.smoke else args.num_workers,
        on_the_fly_dataloading=True,          # mandatory for large volumes (avoids OOM)
        aug_prob_to_one=not args.no_aug,
        iters_per_epoch=args.iters_per_epoch,
    )
    model = SemanticSegmentationUnet(
        max_epochs=args.max_epochs,
        use_deep_supervision=not args.no_deep_supervision,
    )

    ckpt_name = f"{args.project_name}_{args.sub_name}"
    callbacks = [
        ModelCheckpoint(dirpath=args.ckpt_dir, monitor="val_loss", mode="min",
                        save_top_k=3, filename=ckpt_name + "-{epoch:02d}-{val_loss:.3f}"),
        ModelCheckpoint(dirpath=args.ckpt_dir, every_n_epochs=10, save_top_k=-1,
                        filename=ckpt_name + "-{epoch:02d}"),
        LearningRateMonitor(logging_interval="epoch"),
        PrintLearningRate(),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        max_epochs=1 if args.smoke else args.max_epochs,
        limit_train_batches=3 if args.smoke else args.iters_per_epoch,
        limit_val_batches=2 if args.smoke else 1.0,
        num_sanity_val_steps=0 if args.smoke else 2,
        logger=False if args.smoke else [pl_loggers.CSVLogger(args.log_dir)],
        callbacks=None if args.smoke else callbacks,
        enable_checkpointing=not args.smoke,
    )
    trainer.fit(model, dm)

    if args.smoke:
        import math
        tl = trainer.callback_metrics.get("train_loss")
        vd = trainer.callback_metrics.get("val_dice")
        tl_f = float(tl) if tl is not None else float("nan")
        print(f"\n[smoke] train_loss={tl_f}  val_dice={float(vd) if vd is not None else None}")
        assert math.isfinite(tl_f), (
            "SMOKE TEST FAILED: train_loss is NaN/inf. Most common cause: a label crop that is "
            "entirely ignore(2) or otherwise no valid voxels -> loss divides by zero. Check labels."
        )
        print("SMOKE TEST PASSED: train_loss is FINITE; pipeline OK on GPU.")


if __name__ == "__main__":
    main()
