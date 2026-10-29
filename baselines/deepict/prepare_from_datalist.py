#!/usr/bin/env python
"""Build a DeePiCt dataset (mrc + metadata.csv + config.yaml) from a datalist JSON.

JSON: {"training":[{image,label}], "validation":[...], "test":[...]} with paths used verbatim.
Entries may be .pt cubes or .nii.gz/.mrc volumes (auto-detected).

Binary or multi-class, decided from the label values: >1 distinct foreground value -> multi-class.
Use --classes to name the classes (and skip the detection pre-scan), or --binary to force the old
collapse-everything-to-'particle' behaviour.

DeePiCt trains one output channel per semantic class and reads a SEPARATE mask volume for each
(generate_training_data.py), so a multi-class label is split into one binary mask file per class
rather than written as a single multi-valued volume.

Writes under <out>:
  data/raw/<X>.mrc  data/masks/<X>_<class>.mrc (one per class)  data/masks/<X>_region.mrc
  data/metadata.csv  (tomo_name,tomo,lamella_file,<class1>_mask,<class2>_mask,...)
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

from datalist_utils import (binarize, foreground_values, load_datalist, load_volume, remap_12049,
                            save_mrc, split_classes, strip_ext)


def detect_class_values(entries, remap=False):
    """Sorted non-zero label values across the training labels.

    Costs one extra pass over the label volumes, which is why --classes skips it.
    """
    vals = set()
    for i, e in enumerate(entries, 1):
        lab = load_volume(e["label"])
        if remap:
            lab = remap_12049(lab)
        v = foreground_values(lab)
        vals.update(v)
        print(f"  [{i}/{len(entries)}] {strip_ext(e['label']):32s} foreground values={v}", flush=True)
    return sorted(vals)


def convert_entries(entries, raw_d, msk_d, class_names, class_values, remap=False):
    """Write raw + region + one binary mask per class.  class_values is None in binary mode.

    Returns rows of (tomo_name, raw, region, mask_c1, mask_c2, ...) for metadata.csv.
    """
    rows, seen = [], set()
    for i, e in enumerate(entries, 1):
        X = strip_ext(e["label"])
        img = load_volume(e["image"])
        lab = load_volume(e["label"])
        if remap:
            lab = remap_12049(lab)
        raw_p = os.path.join(raw_d, X + ".mrc")
        reg_p = os.path.join(msk_d, X + "_region.mrc")
        save_mrc(img, raw_p)
        save_mrc((img != 0).astype(np.float32), reg_p)

        seen.update(foreground_values(lab))
        if class_values is None:                      # binary: every class collapsed to one mask
            masks = [binarize(lab)]
        else:
            masks = split_classes(lab, class_values)
        mask_ps = []
        for name, m in zip(class_names, masks):
            p = os.path.join(msk_d, f"{X}_{name}.mrc")
            save_mrc(m, p)
            mask_ps.append(p)

        rows.append(tuple([X, raw_p, reg_p] + mask_ps))
        counts = {n: int(m.sum()) for n, m in zip(class_names, masks)}
        print(f"  [{i}/{len(entries)}] {X:32s} shape={img.shape}  fg voxels={counts}", flush=True)
    return rows, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", required=True)
    ap.add_argument("--out", required=True, help="run dir (holds data/, work/, out/, config.yaml)")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "config.template.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--iters-per-epoch", type=int, default=300)
    ap.add_argument("--include-val-in-prediction", action="store_true")
    ap.add_argument("--classes", default=None,
                    help="comma-separated class names for multi-class, in label-value order "
                         "(label value i+1 -> i-th name), e.g. 'ribosome,membrane,fas'. "
                         "Giving this skips the label pre-scan.")
    ap.add_argument("--binary", action="store_true",
                    help="force the old binary behaviour (all classes collapsed to 'particle') "
                         "even when the labels are multi-class")
    ap.add_argument("--remap-12049", action="store_true",
                    help="apply Dataset049/EMPIAR-12049's background merge + renumbering (raw "
                         "{0..5} -> {0,1,2,3}, matching augmentations.py's _remap_12049) to every "
                         "label before class detection/splitting. Use for Dataset049 only.")
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

    # ---- decide binary vs multi-class -------------------------------------------------------
    if args.binary:
        class_names, class_values = ["particle"], None
        print("== mode: binary (--binary) ==")
    elif args.classes:
        class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
        class_values = list(range(1, len(class_names) + 1))   # value i+1 -> i-th name
        print(f"== mode: multi-class, {len(class_names)} classes {class_names} "
              f"(--classes given, skipping label pre-scan) ==")
    else:
        print("== scanning training labels to detect classes ==")
        vals = detect_class_values(d["training"], remap=args.remap_12049)
        if len(vals) <= 1:
            class_names, class_values = ["particle"], None
            print(f"== mode: binary (foreground values {vals}) ==")
        else:
            class_values = vals
            class_names = [f"class{v}" for v in vals]
            print(f"== mode: multi-class, {len(vals)} classes, values {vals} -> {class_names} ==")
            print("   (pass --classes name1,name2,... to give them real names: the names end up in "
                  "metadata.csv columns, output paths and metrics)")

    print("== training -> mrc ==")
    train_rows, seen_tr = convert_entries(d["training"], raw_d, msk_d, class_names, class_values,
                                           remap=args.remap_12049)
    test_entries = d["test"] + (d["validation"] if args.include_val_in_prediction else [])
    print("== test -> mrc ==")
    test_rows, seen_te = convert_entries(test_entries, raw_d, msk_d, class_names, class_values,
                                          remap=args.remap_12049)
    seen = seen_tr | seen_te

    # ---- validate what we actually saw -------------------------------------------------------
    if class_values is None and len(seen) > 1:
        print(f"\nWARNING: labels contain {len(seen)} foreground values {sorted(seen)} but this run "
              f"is BINARY -- every class was collapsed into one 'particle' mask.\n"
              f"         Drop --binary (or pass --classes) to train them as separate classes.\n")
    elif class_values is not None:
        missing = sorted(set(class_values) - seen)
        extra = sorted(seen - set(class_values))
        if extra:
            raise SystemExit(
                f"ERROR: labels contain foreground value(s) {extra} with no class assigned. "
                f"{len(class_names)} classes were declared ({class_names}) covering values "
                f"{class_values}. Pass --classes with one name per label value.")
        if missing:
            print(f"\nWARNING: declared class value(s) {missing} never appear in any label; their "
                  f"masks are all zero.\n")

    meta = os.path.join(data, "metadata.csv")
    with open(meta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tomo_name", "tomo", "lamella_file"] + [f"{n}_mask" for n in class_names])
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
    cfg["training"]["semantic_classes"] = class_names
    cfg["training"]["iterations_per_epoch"] = args.iters_per_epoch
    cfg["training"]["unet_hyperparameters"]["epochs"] = args.epochs
    cfg["prediction"]["active"] = True
    # predict.sh loops every class; this is just the default single-class entry
    cfg["prediction"]["semantic_class"] = class_names[0]
    cfg_path = os.path.join(args.out, "config.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)

    print(f"\nmetadata.csv -> {meta}  (train={len(train_rows)} test={len(test_rows)})")
    print(f"config.yaml  -> {cfg_path}")
    print(f"classes      -> {class_names}")


if __name__ == "__main__":
    main()
