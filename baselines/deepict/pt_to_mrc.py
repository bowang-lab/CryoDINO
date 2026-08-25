#!/usr/bin/env python
"""EXAMPLE converter: 512^3 .pt cubes -> .mrc for DeePiCt (binary 'particle').

Dataset-specific (assumes the CZII .pt layout: images/<b>.pt density, labels/<b>.pt class ids).
For a different dataset, write your own raw + per-class-mask .mrc files and a metadata.csv with
columns  tomo_name, tomo(raw .mrc), <region_col>, <class>_mask .  Everything downstream is generic.

Outputs under --dst:
  raw/<b>.mrc            float32 density        (the 'tomo' column)
  masks/<b>_particle.mrc float32 0/1 = label>0 (the 'particle_mask' column)
  masks/<b>_region.mrc   float32 0/1 = img!=0  (the 'lamella_file' region column)
  metadata.csv           tomo_name,tomo,lamella_file,particle_mask
and prints a tomogram-based train/val split (whole tomograms to one side).
"""
import argparse, csv, os, re
import numpy as np, torch, mrcfile

VAL_TOMOS = {"TE12_0000", "TF6_0000", "UE10_0000", "UF2_0000"}  # held out for prediction_list


def tomo_id(name): return re.sub(r"_patch_.*$", "", name)


def save_mrc(arr, path):
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(arr, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    # --src is DATASET-SPECIFIC: the CZII .pt-cube dataset. Point it at your own copy.
    ap.add_argument("--src", default="/home/cyyu/projects/def-wanglab-ab/cryodino/010/"
                                     "Dataset010_CZII_10010_train_patches_512")
    ap.add_argument("--dst", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--only-tomos", default="", help="comma list of tomo_ids; empty = all")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    raw_d, msk_d = os.path.join(args.dst, "raw"), os.path.join(args.dst, "masks")
    os.makedirs(raw_d, exist_ok=True); os.makedirs(msk_d, exist_ok=True)

    names = sorted(f[:-3] for f in os.listdir(os.path.join(args.src, "images")) if f.endswith(".pt"))
    if args.only_tomos:
        keep = set(args.only_tomos.split(",")); names = [n for n in names if tomo_id(n) in keep]
    if args.limit:
        names = names[:args.limit]

    rows = []
    for i, name in enumerate(names, 1):
        img = torch.load(os.path.join(args.src, "images", name + ".pt"), map_location="cpu").numpy().astype(np.float32)
        lab = torch.load(os.path.join(args.src, "labels", name + ".pt"), map_location="cpu").numpy()
        raw_p = os.path.join(raw_d, name + ".mrc")
        part_p = os.path.join(msk_d, name + "_particle.mrc")
        reg_p = os.path.join(msk_d, name + "_region.mrc")
        save_mrc(img, raw_p)
        save_mrc((lab > 0).astype(np.float32), part_p)
        save_mrc((img != 0).astype(np.float32), reg_p)
        rows.append((name, raw_p, reg_p, part_p))
        print(f"[{i}/{len(names)}] {name:32s} {tomo_id(name):12s} fg={int((lab>0).sum())}", flush=True)

    with open(os.path.join(args.dst, "metadata.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tomo_name", "tomo", "lamella_file", "particle_mask"]); w.writerows(rows)

    train = [n for n, *_ in rows if tomo_id(n) not in VAL_TOMOS]
    val = [n for n, *_ in rows if tomo_id(n) in VAL_TOMOS]
    print(f"\nmetadata.csv: {os.path.join(args.dst,'metadata.csv')}  ({len(rows)} rows)")
    print(f"training_list ({len(train)}): {train}")
    print(f"prediction_list ({len(val)}): {val}")


if __name__ == "__main__":
    main()
