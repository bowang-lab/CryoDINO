#!/usr/bin/env python
"""Build a DeePiCt dataset (mrc + metadata.csv + config.yaml) from a datalist JSON.

JSON: {"training":[{image,label}], "validation":[...], "test":[...]} with paths used verbatim.
Entries may be .pt cubes or .nii.gz/.mrc volumes (auto-detected). Binary 'particle' target.

Writes under <out>:
  data/raw/<X>.mrc  data/masks/<X>_particle.mrc  data/masks/<X>_region.mrc   (train + test)
  data/metadata.csv  (tomo_name,tomo,lamella_file,particle_mask)
  config.yaml        (training_list = train names, prediction_list = test names,
                      training.active + prediction.active = true, binary 'particle')
DeePiCt derives its own val from train_split of the training boxes, so the JSON 'validation'
split is not used by training (kept out; pass --include-val-in-prediction to also predict it).
X = basename of the entry's label with extension stripped.
"""
import argparse
import csv
import os

import numpy as np
import yaml

from datalist_utils import binarize, load_datalist, load_volume, save_mrc, strip_ext


def convert_entries(entries, raw_d, msk_d):
    rows = []
    for i, e in enumerate(entries, 1):
        X = strip_ext(e["label"])
        img = load_volume(e["image"])
        raw_p = os.path.join(raw_d, X + ".mrc")
        part_p = os.path.join(msk_d, X + "_particle.mrc")
        reg_p = os.path.join(msk_d, X + "_region.mrc")
        save_mrc(img, raw_p)
        save_mrc(binarize(load_volume(e["label"])), part_p)
        save_mrc((img != 0).astype(np.float32), reg_p)
        rows.append((X, raw_p, reg_p, part_p))
        print(f"  [{i}/{len(entries)}] {X:32s} shape={img.shape}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", required=True)
    ap.add_argument("--out", required=True, help="run dir (holds data/, work/, out/, config.yaml)")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "config.template.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--iters-per-epoch", type=int, default=300)
    ap.add_argument("--include-val-in-prediction", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print resolved file lists, convert nothing")
    args = ap.parse_args()

    d = load_datalist(args.datalist)
    if args.dry_run:
        for split in ("training", "validation", "test"):
            print(f"\n[{split}] n={len(d[split])}")
            for e in d[split]:
                print(f"  {strip_ext(e['label']):32s} img={e['image']}")
        return
    data = os.path.join(args.out, "data")
    raw_d, msk_d = os.path.join(data, "raw"), os.path.join(data, "masks")
    os.makedirs(raw_d, exist_ok=True); os.makedirs(msk_d, exist_ok=True)

    print(f"datalist: training={len(d['training'])} validation={len(d['validation'])} test={len(d['test'])}")
    print("== training -> mrc ==")
    train_rows = convert_entries(d["training"], raw_d, msk_d)
    test_entries = d["test"] + (d["validation"] if args.include_val_in_prediction else [])
    print("== test -> mrc ==")
    test_rows = convert_entries(test_entries, raw_d, msk_d)

    meta = os.path.join(data, "metadata.csv")
    with open(meta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tomo_name", "tomo", "lamella_file", "particle_mask"])
        w.writerows(train_rows + test_rows)

    cfg = yaml.safe_load(open(args.template))
    cfg["dataset_table"] = meta
    cfg["output_dir"] = os.path.join(args.out, "out")
    cfg["work_dir"] = os.path.join(args.out, "work")
    cfg["model_path"] = os.path.join(args.out, "out", "model.pth")
    cfg["cluster"]["logdir"] = os.path.join(args.out, "logs")
    cfg["tomos_sets"]["training_list"] = [r[0] for r in train_rows]
    cfg["tomos_sets"]["prediction_list"] = [r[0] for r in test_rows]
    cfg["training"]["active"] = True
    cfg["training"]["semantic_classes"] = ["particle"]
    cfg["training"]["iterations_per_epoch"] = args.iters_per_epoch
    cfg["training"]["unet_hyperparameters"]["epochs"] = args.epochs
    cfg["prediction"]["active"] = True
    cfg["prediction"]["semantic_class"] = "particle"
    cfg_path = os.path.join(args.out, "config.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)

    print(f"\nmetadata.csv -> {meta}  (train={len(train_rows)} test={len(test_rows)})")
    print(f"config.yaml  -> {cfg_path}")


if __name__ == "__main__":
    main()
