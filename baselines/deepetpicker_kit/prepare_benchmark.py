#!/usr/bin/env python
"""Build a DeepETPicker base_path for the CryoDINO detection benchmark split.

prepare_from_datalist.py only uses the datalist's `training` patches and holds two of those
tomograms out as val/test. The benchmark instead trains on Dataset440 (the `training` patches,
reassembled), validates on the `validation` tomograms (Dataset445) and tests on the `test`
tomograms (Dataset446), the same split detection3d.py uses.

Layout written under <out>:
  raw_data/0tr_<tomo>.mrc/.coords   training tomograms, reassembled from the patches
  raw_data/1va_<tomo>.mrc/.coords   the ONE val tomogram DeepETPicker's own val loss runs on
                                    (train_bash.py only ever validates on a single tomogram)
  data_std/{1va,2te}_<tomo>.mrc     every val/test tomogram, standardised exactly as
                                    utils/normalization.py does (x - mean) / std, written here
                                    directly: preprocess.py then only handles raw_data/, and its
                                    Pool(len(files)) would otherwise fork one process per tomogram
  gt/<name>.npy                     (N, 4) float rows (x, y, z, class) for scoring
  dataset_split.json                names, roles, ids, radii, voxel size
  <dset>_preprocess.py              DeepETPicker preprocessing config

The 0tr_/1va_/2te_ prefixes keep DeepETPicker's dataset ids (sorted file names) as training
ids 0..n-1 followed by the val id: TS_100_3 would otherwise sort before TS_5_4.

Axes: a val/test .nii.gz loads (nibabel) as V[x, y, z], the same order the patches index
(see prepare_from_datalist.py), so it is written as V.transpose(2, 1, 0) and its points
[p0, p1, p2, cls, r] stay (x, y, z) = (p0, p1, p2). --check-orientation compares one
reassembled training tomogram with its original .nii.gz to confirm this.
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from datalist_utils import load_datalist, save_mrc, strip_ext
from prepare_from_datalist import assemble, group_by_tomogram, merge_points, odd_up

PREFIX_TRAIN, PREFIX_VAL, PREFIX_TEST = "0tr_", "1va_", "2te_"


def tomo_name(image_path):
    """'.../imagesTr/TS_100_3_0000.nii.gz' -> 'TS_100_3'."""
    name = strip_ext(image_path)
    return name[:-5] if name.endswith("_0000") else name


def load_nii(path):
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)


def standardise(vol):
    """utils/normalization.py, norm_type='standardization', on the float32 volume."""
    vol = vol.astype(np.float32)
    vol -= vol.mean()
    vol /= vol.std()
    return vol


def gt_rows(points):
    """Datalist points [x, y, z, cls, r] -> (N, 4) rows (x, y, z, cls), sentinels dropped."""
    rows = [[p[0], p[1], p[2], p[3]] for p in points if p[3] >= 0]
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def write_std(job):
    """Worker: one val/test tomogram -> data_std/<name>.mrc (skipped if already there)."""
    image, dst = job
    if os.path.exists(dst):
        return dst, "reused"
    vol = standardise(load_nii(image))
    tmp = dst + ".tmp"
    save_mrc(vol.transpose(2, 1, 0), tmp)
    os.replace(tmp, dst)
    return dst, "written"


def check_orientation(train_vol, nii_path):
    """Correlation between a reassembled training tomogram and its .nii.gz, both standardised."""
    ref = load_nii(nii_path)
    if ref.shape != train_vol.shape:
        raise SystemExit(f"orientation check: shape {train_vol.shape} vs {ref.shape} ({nii_path})")
    a, b = standardise(train_vol.copy()).ravel(), standardise(ref).ravel()
    corr = float(np.dot(a, b) / a.size)
    print(f"orientation check vs {os.path.basename(nii_path)}: corr={corr:.4f}")
    if corr < 0.9:
        raise SystemExit("orientation check failed: the patches and the .nii.gz disagree")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-root", default=None,
                    help="dir holding the .pt patches (default: the paths recorded in the datalist)")
    ap.add_argument("--dset-name", default="Dataset440_preprocess")
    ap.add_argument("--label-diameter", type=int, default=9)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--check-orientation", default=None,
                    help="dir holding the original training .nii.gz (<tomo>_0000.nii.gz)")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    raw, std, gt = (os.path.join(out, d) for d in ("raw_data", "data_std", "gt"))
    for d in (raw, std, gt):
        os.makedirs(d, exist_ok=True)

    d = load_datalist(args.datalist)
    print("datalist: " + " ".join(f"{k}={len(d[k])}" for k in ("training", "validation", "test")))
    if not d["validation"] or not d["test"]:
        raise SystemExit("the benchmark split needs non-empty validation and test")
    image_root = args.image_root   # None -> load_patch uses each entry's own path

    # ---- 1) training tomograms, reassembled from the patches (labels: .coords)
    groups, skipped = group_by_tomogram(d["training"])
    if skipped:
        raise SystemExit(f"{len(skipped)} training entries are not patches (e.g. {skipped[0]})")
    train_tomos = sorted(groups)
    radii = {}
    for tomo in train_tomos:
        rows, r = merge_points(groups[tomo])
        for c, v in r.items():
            radii.setdefault(c, v)
        name = PREFIX_TRAIN + tomo
        mrc = os.path.join(raw, name + ".mrc")
        if not os.path.exists(mrc):
            vol = assemble(groups[tomo], image_root)
            if args.check_orientation and tomo == train_tomos[0]:
                check_orientation(vol, os.path.join(args.check_orientation, tomo + "_0000.nii.gz"))
            save_mrc(vol.transpose(2, 1, 0), mrc)
        coords = np.array([[c + 1, x, y, z] for x, y, z, c in rows])
        np.savetxt(os.path.join(raw, name + ".coords"), np.rint(coords).astype(int),
                   fmt="%d", delimiter="\t")
        print(f"  train {name}: {len(rows)} particles", flush=True)

    # ---- 2) val/test tomograms: GT for scoring + standardised volumes in data_std/
    val, test = d["validation"], d["test"]
    roles, jobs = {}, []
    for prefix, role, entries in ((PREFIX_VAL, "val", val), (PREFIX_TEST, "test", test)):
        for e in entries:
            name = prefix + tomo_name(e["image"])
            if name in roles:
                raise SystemExit(f"duplicate tomogram name {name}")
            roles[name] = role
            np.save(os.path.join(gt, name + ".npy"), gt_rows(e["points"]))
            jobs.append((e["image"], os.path.join(std, name + ".mrc")))
    voxel_sizes = {e.get("voxel_size") for e in val + test}
    if len(voxel_sizes) != 1:
        raise SystemExit(f"val/test voxel sizes differ: {voxel_sizes}")

    # The single tomogram DeepETPicker's own val loss (checkpoint choice) runs on: the first
    # val entry. It needs raw_data + .coords so preprocess.py builds its labels; preprocess.py
    # also writes its data_std copy, so it is taken out of the direct data_std jobs.
    dep_val = PREFIX_VAL + tomo_name(val[0]["image"])
    jobs = [j for j in jobs if os.path.basename(j[1]) != dep_val + ".mrc"]
    dep_val_mrc = os.path.join(raw, dep_val + ".mrc")
    if not os.path.exists(dep_val_mrc):
        save_mrc(load_nii(val[0]["image"]).transpose(2, 1, 0), dep_val_mrc)
    rows = gt_rows(val[0]["points"])
    coords = np.column_stack([rows[:, 3] + 1, rows[:, 0], rows[:, 1], rows[:, 2]])
    np.savetxt(os.path.join(raw, dep_val + ".coords"), np.rint(coords).astype(int),
               fmt="%d", delimiter="\t")

    print(f"standardising {len(jobs)} val/test tomograms into {std} ({args.workers} workers)",
          flush=True)
    with Pool(args.workers) as p:
        for i, (dst, how) in enumerate(p.imap_unordered(write_std, jobs), 1):
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ({how}: {os.path.basename(dst)})", flush=True)

    # ---- 3) split record + preprocessing config
    classes = sorted(radii)
    diameters = [odd_up(2 * radii[c]) for c in classes]
    names = [PREFIX_TRAIN + t for t in train_tomos] + [dep_val]   # == DeepETPicker ids
    split = {
        "tomograms": names,
        "train_ids": list(range(len(train_tomos))),
        "val_id": len(train_tomos),
        "roles": roles,                                  # every scored tomogram -> val/test
        "voxel_size": voxel_sizes.pop(),
        "classes": classes,
        "radii_voxels": {str(c): radii[c] for c in classes},
        "ocp_diameter": ",".join(str(x) for x in diameters),
        "num_cls": len(classes),
        "label_diameter": args.label_diameter,
        "datalist": os.path.abspath(args.datalist),
    }
    with open(os.path.join(out, "dataset_split.json"), "w") as f:
        json.dump(split, f, indent=2)

    if args.label_diameter >= min(diameters) or args.label_diameter % 2 == 0:
        raise SystemExit(f"--label-diameter must be odd and < {min(diameters)}")
    pre_config = {
        "dset_name": args.dset_name,
        "base_path": out,
        "coord_path": raw,
        "coord_format": ".coords",
        "tomo_path": raw,
        "tomo_format": ".mrc",
        "num_cls": len(classes),
        "label_type": "gaussian",
        "label_diameter": args.label_diameter,
        "ocp_type": "sphere",
        "ocp_diameter": split["ocp_diameter"],
        "norm_type": "standardization",
    }
    with open(os.path.join(out, f"{args.dset_name}.py"), "w") as f:
        f.write("pre_config=")
        json.dump(pre_config, f, indent=12)

    n_val = sum(r == "val" for r in roles.values())
    print(f"train ids 0-{len(train_tomos) - 1}: {names[:-1]}")
    print(f"DeepETPicker val id {split['val_id']}: {dep_val}")
    print(f"scored: {n_val} val, {len(roles) - n_val} test")
    # consumed by run_benchmark.sh
    print(f"TRAIN_SET_IDS=0-{len(train_tomos) - 1}")
    print(f"VAL_SET_IDS={split['val_id']}")


if __name__ == "__main__":
    main()
