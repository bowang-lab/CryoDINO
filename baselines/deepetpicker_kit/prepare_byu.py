#!/usr/bin/env python
"""BYU (flagellar motor) version of prepare_from_datalist.py: build a DeepETPicker base_path.

The BYU datalists give 128^3 .pt patches for training (named tomo_<id>_<i>_<j>_<k>, points in
patch coordinates) and whole .nii.gz tomograms for validation/test. The patches do not tile each
tomogram (median ~10-17 per tomogram), so unlike CZI the tomograms cannot be re-assembled from
them. Instead the whole training tomograms are read from --tomo-dir (the folder the validation
tomograms live in), and the patch points are moved to tomogram coordinates (point + offset).

Why each training tomogram is CROPPED around its motors
  DeepETPicker trains on block_size crops centred on each particle, shifted by at most
  block_size//5 (dataloader_DynamicLoad.__sample, random_num=0 = its default and our CZI setting),
  so it never reads a voxel further than block_size//2 + block_size//5 (= 50 for block 72) from a
  motor. Whole BYU tomograms are ~1 GB each as float32 and DeepETPicker loads every training
  tomogram into RAM (plus a label and an occupancy volume of the same size), which does not fit
  for 253-452 tomograms. A box of motors +- --crop-margin voxels (clipped to the tomogram) holds
  every voxel training can touch, so the training samples are the same as with whole tomograms.
  run_benchmark.sh checks margin >= block//2 + block//5.

Why the crop is written already standardised
  DeepETPicker standardises each raw_data volume with its own mean/std (utils/normalization.py).
  For a crop that would use the crop's statistics, not the tomogram's, unlike inference
  (infer_splits.py standardises whole tomograms). So the crop is cut from the whole-tomogram-
  standardised volume -- the same float32 ops as datalist_utils.standardize -- and
  run_benchmark.sh copies raw_data/*.mrc over data_std/*.mrc after preprocess.py
  (dataset_split.json: "prestandardized": true).

Motor-free training tomograms carry no particle, so DeepETPicker draws no sample from them
(random_num=0); they are skipped and counted. Motor size in voxels varies with the voxel size
(13.1-19.7 A); DeepETPicker takes ONE diameter per class, so the median training radius is used
(override: --diameter).

    python prepare_byu.py --datalist <byu datalist> --out <base_path> --internal-val 0
"""
import argparse
import collections
import json
import math
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

from datalist_utils import (check_points_in_volume, load_datalist, load_nifti_xyz, load_patch,
                            resolve_path, save_mrc, strip_ext, valid_points)
from prepare_from_datalist import INTERNAL_VAL_PREFIX, odd_up, pick_internal_val

PATCH = 128
BYU_PATCH_RE = re.compile(r"^(tomo_[0-9a-zA-Z]+)_(\d+)_(\d+)_(\d+)$")
MERGE_TOL = 1.0  # voxels: the same motor seen from overlapping patches


def group_patches(entries):
    groups = collections.defaultdict(list)
    for e in entries:
        m = BYU_PATCH_RE.match(strip_ext(e["image"]))
        if m is None:
            raise SystemExit(f"not a BYU patch name: {e['image']}")
        off = tuple(int(m.group(i)) for i in (2, 3, 4))
        groups[m.group(1)].append((off, e.get("points", []), e["image"]))
    return groups


def merge_points(items):
    """[(offset, points)] -> unique tomogram-coordinate rows (x, y, z, cls, radius)."""
    rows = []
    for off, points in items:
        for p in valid_points(points):
            g = np.array([p[0] + off[0], p[1] + off[1], p[2] + off[2], p[3], p[4]])
            if not any(r[3] == g[3] and np.abs(r[:3] - g[:3]).max() <= MERGE_TOL for r in rows):
                rows.append(g)
    return rows


def prepare_one(job):
    """Load one whole tomogram, verify patches against it, write the standardised crop."""
    name, path, rows, patches, args = job
    mrc = os.path.join(args.out, "raw_data", name + ".mrc")
    side = os.path.join(args.out, "crops", name + ".json")
    if os.path.exists(mrc) and os.path.exists(side) and not args.force:
        info = json.load(open(side))
        info["reused"] = True
        return info

    vol = load_nifti_xyz(path)                              # (X, Y, Z) float32, as at inference
    shape = vol.shape
    pts = np.array([r[:3] for r in rows])
    try:
        check_points_in_volume([list(r) for r in rows], shape, name)
    except SystemExit as e:              # a SystemExit would hang the Pool
        raise RuntimeError(str(e))
    for off, _, ppath in patches[: args.verify_patches]:
        cube = load_patch(ppath)
        ref = vol[off[0]:off[0] + PATCH, off[1]:off[1] + PATCH, off[2]:off[2] + PATCH]
        # CryoDINO's patches may be raw or z-scored (augmentations.py says z-scored), so compare
        # up to an affine intensity map: a wrong offset/axis order gives correlation ~0
        same = cube.shape == ref.shape and np.corrcoef(cube.ravel(), ref.ravel())[0, 1]
        if not same or same < 0.999:
            raise RuntimeError(f"{name}: patch {os.path.basename(ppath)} != tomogram crop at offset "
                             f"{off} (shapes {cube.shape} vs {ref.shape}, corr {same}) -- the "
                             f"offset/axis convention differs from CZI's; points would be wrong")

    lo = np.maximum(0, np.floor(pts.min(0) - args.crop_margin)).astype(int)
    hi = np.minimum(shape, np.ceil(pts.max(0) + args.crop_margin) + 1).astype(int)
    # A crop cut short by a tomogram edge must still be >= 2*margin+1 (>= block_size) along that
    # axis, or DeepETPicker's edge clamp (__sample) would place the sample differently than in the
    # whole tomogram: grow it into the tomogram's interior.
    need = np.minimum(shape, 2 * args.crop_margin + 1)
    short = hi - lo < need
    hi = np.where(short & (lo == 0), need, hi)
    lo = np.where(short & (hi == np.array(shape)), np.array(shape) - need, lo)
    vol -= vol.mean()                     # same ops as datalist_utils.standardize (whole tomogram)
    s = vol.std()
    crop = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] / s
    del vol
    save_mrc(crop.transpose(2, 1, 0), mrc)
    coords = np.array([[int(r[3]) + 1, *(r[:3] - lo)] for r in rows])
    np.savetxt(os.path.join(args.out, "raw_data", name + ".coords"), np.rint(coords).astype(int),
               fmt="%d", delimiter="\t", newline="\n")
    info = {"name": name, "source": path, "tomo_shape_xyz": list(shape), "crop_lo_xyz": lo.tolist(),
            "crop_shape_xyz": list(crop.shape), "n": len(rows),
            "radii": sorted({round(float(r[4]), 3) for r in rows})}
    with open(side, "w") as f:
        json.dump(info, f)
    info["reused"] = False
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--internal-val", default="0",
                    help="index or basename of the `validation` tomogram used for DeepETPicker's "
                         "val_loss (must contain a motor), or 'all': every validation tomogram "
                         "with a motor (needs DEP_FULL_VAL=1 at training, see patch_periodic_ckpt.py)")
    ap.add_argument("--tomo-dir", default=None,
                    help="folder with the whole training tomograms <tomo>.nii.gz (default: the "
                         "folder of the first validation tomogram)")
    ap.add_argument("--nii-root", default=None, help="fallback dir for validation .nii.gz paths")
    ap.add_argument("--crop-margin", type=int, default=64)
    ap.add_argument("--diameter", type=int, default=None,
                    help="motor diameter in voxels (odd); default: odd_up(2 * median radius)")
    ap.add_argument("--verify-patches", type=int, default=1,
                    help="patches per tomogram compared voxel-for-voxel with the tomogram")
    ap.add_argument("--workers", type=int, default=3, help="tomograms loaded in parallel "
                    "(~2x tomogram size in RAM each: 2 GB typical, up to ~14 GB)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)

    d = load_datalist(args.datalist)
    print("datalist: " + " ".join(f"{k}={len(d[k])}" for k in ("training", "validation", "test")))
    groups = group_patches(d["training"])
    tomo_dir = args.tomo_dir or os.path.dirname(resolve_path(d["validation"][0]["image"], args.nii_root))

    jobs, skipped = [], []
    for t in sorted(groups):
        rows = merge_points([(off, pts) for off, pts, _ in groups[t]])
        if not rows:
            skipped.append(t)
            continue
        path = os.path.join(tomo_dir, t + ".nii.gz")
        if not os.path.exists(path):
            raise SystemExit(f"whole training tomogram missing: {path} (set --tomo-dir)")
        jobs.append([t, path, rows, groups[t]])
    print(f"{len(groups)} training tomograms: {len(jobs)} with a motor, {len(skipped)} motor-free "
          f"(no DeepETPicker samples, skipped)")

    n_tr = len(jobs)
    if args.internal_val == "all":
        # every validation tomogram with a motor; motor-free ones give DeepETPicker no val sample
        val_entries = [(e, strip_ext(os.path.basename(e["image"]))) for e in d["validation"]]
    else:
        val_entries = [pick_internal_val(d["validation"], args.internal_val)]
    for val_entry, val_src in val_entries:
        val_rows = merge_points([((0, 0, 0), val_entry.get("points", []))])
        if not val_rows:
            if args.internal_val == "all":
                continue
            raise SystemExit(f"internal val {val_src} has no motor -- pick another --internal-val")
        jobs.append([INTERNAL_VAL_PREFIX + val_src, resolve_path(val_entry["image"], args.nii_root),
                     val_rows, []])
    val_names = [j[0] for j in jobs[n_tr:]]
    if sorted(j[0] for j in jobs)[n_tr:] != sorted(val_names):
        raise SystemExit(f"internal val tomograms would not sort after the training tomograms")
    jobs = jobs[:n_tr] + sorted(jobs[n_tr:], key=lambda j: j[0])
    print(f"internal val: {len(jobs) - n_tr} tomogram(s) -> dataset ids {n_tr}-{len(jobs) - 1}")

    radii = [float(r[4]) for j in jobs[:n_tr] for r in j[2]]
    diameter = args.diameter or odd_up(2 * float(np.median(radii)))
    if diameter % 2 == 0:
        raise SystemExit("--diameter must be odd (DeepETPicker coords2labels.py)")
    print(f"motor radius in voxels over {len(radii)} training motors: min {min(radii):.1f} "
          f"median {np.median(radii):.1f} max {max(radii):.1f} -> diameter {diameter}")
    if args.dry_run:
        return

    os.makedirs(os.path.join(args.out, "raw_data"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)
    infos = {}
    with Pool(args.workers) as pool:
        for k, info in enumerate(pool.imap_unordered(prepare_one, [tuple(j) + (args,) for j in jobs]), 1):
            infos[info["name"]] = info
            print(f"  [{k}/{len(jobs)}] {info['name']} tomo {tuple(info['tomo_shape_xyz'])} -> crop "
                  f"{tuple(info['crop_shape_xyz'])} motors {info['n']}"
                  + (" (reused)" if info["reused"] else ""), flush=True)

    names = [j[0] for j in jobs]
    n_train = sum(infos[n]["n"] for n in names[:n_tr])
    mb = sum(np.prod(infos[n]["crop_shape_xyz"]) for n in names) * 4 / 1e6
    split = {
        "tomograms": names,
        "particle_counts": {n: infos[n]["n"] for n in names},
        "volume_shape_xyz": {n: infos[n]["crop_shape_xyz"] for n in names},
        "crop_lo_xyz": {n: infos[n]["crop_lo_xyz"] for n in names},
        "classes": [0],
        "radii_voxels": {"0": float(np.median(radii))},
        "diameters_voxels": {"0": diameter},
        "ocp_diameter": str(diameter),
        "num_cls": 1,
        "raw_data": os.path.join(args.out, "raw_data"),
        "prestandardized": True,
        "crop_margin": args.crop_margin,
        "n_train_particles": n_train,
        "skipped_motor_free": skipped,
        "val_id": n_tr, "val_ids": list(range(n_tr, len(names))), "test_id": None,
        "internal_val_source": [n[len(INTERNAL_VAL_PREFIX):] for n in names[n_tr:]],
        "internal_val_image": [j[1] for j in jobs[n_tr:]],
    }
    with open(os.path.join(args.out, "dataset_split.json"), "w") as f:
        json.dump(split, f, indent=2)
    print(f"\n{n_tr} training tomograms, {n_train} training motors, {len(names) - n_tr} internal val "
          f"tomogram(s); crops {mb:.0f} MB "
          f"float32 in raw_data (x~4 after preprocess)")
    print(f"split written: {os.path.join(args.out, 'dataset_split.json')}")


if __name__ == "__main__":
    sys.exit(main())
