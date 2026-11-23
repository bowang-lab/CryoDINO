#!/usr/bin/env python
"""Add three opt-in training options to DeepETPicker's train.py (idempotent), all off by default.

1. Periodic checkpoints ($DEP_SAVE_EVERY epochs). DeepETPicker keeps one checkpoint
   (ModelCheckpoint save_top_k=1, monitor='val_loss'). To select the checkpoint by
   CZIDetectionMetrics F4 on the full validation split -- as detection3d.py does for CryoDINO --
   we need intermediate checkpoints: version_N/checkpoints_periodic/periodic-epoch=XX.ckpt.

2. Fixed iterations per epoch ($DEP_LIMIT_TRAIN_BATCHES). DeepETPicker's epoch is one pass over
   the particle-centred samples (1264 at 100% -> 316 batches at batch 4). Lightning's
   limit_train_batches=300 stops each (shuffled) epoch after 300 batches, which matches 3DINO's
   --epoch-length 300 exactly: 100 epochs x 300 = 30,000 iterations.

3. Repeated training set ($DEP_REPEAT_TRAIN copies). At batch 8 an epoch has only 158 batches, so
   300 iterations per epoch needs a longer epoch: the loader shuffles over N copies of the
   particle-centred samples and limit_train_batches cuts each epoch at 300. This mirrors 3DINO,
   whose "epoch" is 300 steps drawn from an endless shuffled stream (SHARDED_INFINITE sampler).

4. Validation on every val tomogram ($DEP_FULL_VAL=1). Upstream bin/train_bash.py hard-codes the
   val range to ONE tomogram (data_split = [..., val_first, val_first + 1, ...]), even when
   --val_set_ids lists several. With DEP_FULL_VAL=1 the range covers all of them, so the logged
   val_loss is DeepETPicker's own Dice loss over every motor in all val tomograms (BYU: prepare_byu.py
   --internal-val all). Only the val_loss/val_f1 logging and DeepETPicker's own best-val_loss
   checkpoint see this; the periodic checkpoints and training are unchanged.

Nothing else changes: training loop, loss, LR schedule and DeepETPicker's own best-val_loss
checkpoint stay as they are. With both variables unset (or 0) the patched train.py behaves exactly
like the setup.sh version. Must run after setup.sh (it patches the setup.sh-patched Trainer call).

    python patch_periodic_ckpt.py --dep <DeepETPicker checkout>
"""
import argparse
import os

EDITS = [
    ("DEP_SAVE_EVERY",
     "callbacks=[lr_monitor, checkpoint_callback],",
     "callbacks=[lr_monitor, checkpoint_callback] + ("
     "[ModelCheckpoint(dirpath=os.path.join(tb_logger.log_dir, 'checkpoints_periodic'), "
     "filename='periodic-{epoch:02d}', every_n_epochs=int(os.environ['DEP_SAVE_EVERY']), "
     "save_top_k=-1, save_weights_only=True)] "
     "if int(os.environ.get('DEP_SAVE_EVERY', '0') or 0) > 0 else []),"),
    ("DEP_LIMIT_TRAIN_BATCHES",
     "max_epochs=args.max_epoch,",
     "max_epochs=args.max_epoch,\n"
     "                     limit_train_batches="
     "(int(os.environ.get('DEP_LIMIT_TRAIN_BATCHES', '0') or 0) or 1.0),"),
    ("DEP_REPEAT_TRAIN",
     "        return DataLoader(train_dataset,",
     "        if int(os.environ.get('DEP_REPEAT_TRAIN', '1') or 1) > 1:\n"
     "            train_dataset = torch.utils.data.ConcatDataset("
     "[train_dataset] * int(os.environ['DEP_REPEAT_TRAIN']))\n"
     "        return DataLoader(train_dataset,"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dep", required=True)
    args = ap.parse_args()
    path = os.path.join(args.dep, "train.py")
    t = open(path).read()
    for marker, old, new in EDITS:
        if marker in t:
            print(f"[ok] {path} already has {marker}")
            continue
        if t.count(old) != 1:
            raise SystemExit(f"{path}: expected exactly one '{old}' (the setup.sh-patched Trainer "
                             f"call), found {t.count(old)} -- run setup.sh first")
        t = t.replace(old, new)
        print(f"[ok] patched {path}: {marker}")
    open(path, "w").write(t)

    path = os.path.join(args.dep, "bin", "train_bash.py")
    t = open(path).read()
    old = "val_first, val_first + 1,  # val"
    new = ("val_first, val_first + (len(val_list) if os.environ.get('DEP_FULL_VAL') == '1' else 1),"
           "  # val (DEP_FULL_VAL)")
    if "DEP_FULL_VAL" in t:
        print(f"[ok] {path} already has DEP_FULL_VAL")
    elif t.count(old) != 1:
        raise SystemExit(f"{path}: expected exactly one '{old}', found {t.count(old)}")
    else:
        open(path, "w").write(t.replace(old, new))
        print(f"[ok] patched {path}: DEP_FULL_VAL")


if __name__ == "__main__":
    main()
