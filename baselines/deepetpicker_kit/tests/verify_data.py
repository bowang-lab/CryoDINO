#!/usr/bin/env python
"""Independently verify what prepare + preprocess built, recomputed from the SOURCE datalist.

Nothing here trusts the kit's own bookkeeping: every expected value is derived from the original
`.pt` patches and the datalist JSON, so agreement means the pipeline wrote what it should have.

    python tests/verify_data.py --base <out_dir> --datalist <datalist.json> [--tomos TS_5_4,...]

Stage 1 (raw_data, always) checks the reassembled tomograms and merged coords.
Stage 2 (preprocess) runs only if data_std/ and the label dirs exist.
Exits non-zero on any mismatch.
"""
import argparse
import json
import os
import random
import sys

import mrcfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datalist_utils import load_datalist, parse_patch_name, strip_ext   # noqa: E402

PATCH = 128
FAILS = []


def check(name, cond, detail=""):
    print(f"    {'ok  ' if cond else 'FAIL'}  {name}{'' if cond else '  -- ' + detail}", flush=True)
    if not cond:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the run's out_dir (DeepETPicker base_path)")
    ap.add_argument("--datalist", required=True)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--tomos", default=None, help="comma-separated subset (default: all)")
    ap.add_argument("--patches-per-tomo", type=int, default=3,
                    help="how many random patches to round-trip per tomogram")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    image_root = args.image_root or os.path.join(
        os.path.dirname(os.path.abspath(args.datalist)), "images")
    split = json.load(open(os.path.join(args.base, "dataset_split.json")))
    tomos = [t.strip() for t in args.tomos.split(",")] if args.tomos else split["tomograms"]
    random.seed(args.seed)

    print("dataset ids:", split["tomograms"])
    check("tomogram order is sorted (== DeepETPicker dataset ids)",
          split["tomograms"] == sorted(split["tomograms"]))
    diams = [int(d) for d in split["ocp_diameter"].split(",")]
    check("all ocp diameters odd (coords2labels needs odd edges)",
          all(d % 2 for d in diams), str(diams))

    # group the source patches by tomogram, straight from the datalist
    by_tomo = {}
    for e in load_datalist(args.datalist)["training"]:
        name = strip_ext(e["image"])
        try:
            tomo, off = parse_patch_name(name)
        except ValueError:
            continue
        by_tomo.setdefault(tomo, []).append((name, off, e))

    for tomo in tomos:
        print(f"\n[{tomo}]")
        with mrcfile.open(os.path.join(args.base, "raw_data", f"{tomo}.mrc"),
                          permissive=True) as m:
            vol = np.asarray(m.data)                 # (z, y, x)
        V = vol.transpose(2, 1, 0)                   # back to V[p0, p1, p2]
        exp_shape = tuple(split["volume_shape_xyz"][tomo])
        check(f"mrc shape == reversed {exp_shape}", vol.shape == exp_shape[::-1], str(vol.shape))

        ok = True
        for name, off, _ in random.sample(by_tomo[tomo], args.patches_per_tomo):
            cube = torch.load(os.path.join(image_root, name + ".pt"),
                              map_location="cpu").numpy()
            sl = tuple(slice(off[d], off[d] + PATCH) for d in range(3))
            if not np.array_equal(V[sl], cube):
                ok = False
                check(f"patch {name} round-trips", False, "content differs from the source .pt")
        check(f"{args.patches_per_tomo} random patches round-trip exactly", ok)

        co = np.loadtxt(os.path.join(args.base, "raw_data", f"{tomo}.coords"),
                        delimiter="\t", ndmin=2).astype(int)
        n_cls = split["num_cls"]
        check(f"coords classes within 1..{n_cls}",
              bool(co[:, 0].min() >= 1 and co[:, 0].max() <= n_cls),
              f"{co[:, 0].min()}..{co[:, 0].max()}")
        check("coords inside the volume",
              bool((co[:, 1:] >= 0).all()
                   and (co[:, 1] < exp_shape[0]).all()
                   and (co[:, 2] < exp_shape[1]).all()
                   and (co[:, 3] < exp_shape[2]).all()))
        vals = np.array([V[x, y, z] for _, x, y, z in co])
        check("mrc[z,y,x] == V[x,y,z] at every labelled point",
              np.array_equal(vals, np.array([vol[z, y, x] for _, x, y, z in co])))
        check(f"particles are dark: mean density at points {vals.mean():+.3f} "
              f"<< volume mean {V.mean():+.3f}", vals.mean() < V.mean() - 0.5)

        # recount unique particles from the datalist itself
        uniq = set()
        for _, off, e in by_tomo[tomo]:
            for p in e["points"]:
                if p[3] >= 0:
                    uniq.add(tuple(round(p[d] + off[d], 2) for d in range(3)) + (int(p[3]),))
        check(f"particle count {len(co)} == datalist unique {len(uniq)}", len(co) == len(uniq))

        # ---- stage 2: preprocess outputs, if they exist
        label_dir = os.path.join(args.base, f"gaussian{split.get('label_diameter', 9)}")
        std_p = os.path.join(args.base, "data_std", f"{tomo}.mrc")
        ocp_p = os.path.join(args.base, "data_ocp", f"{tomo}.mrc")
        lab_p = os.path.join(label_dir, f"{tomo}.mrc")
        if not all(os.path.exists(p) and os.path.getsize(p) for p in (std_p, ocp_p, lab_p)):
            print("    --    preprocess outputs absent/empty, skipping stage 2")
            continue
        with mrcfile.open(std_p, permissive=True) as m:
            std = np.asarray(m.data)
        with mrcfile.open(lab_p, permissive=True) as m:
            lab = np.asarray(m.data)
        with mrcfile.open(ocp_p, permissive=True) as m:
            ocp = np.asarray(m.data)
        check(f"data_std standardised (mean {std.mean():+.3f}, sd {std.std():.3f})",
              abs(std.mean()) < 1e-3 and abs(std.std() - 1) < 1e-3)
        check(f"label holds classes 0..{n_cls}",
              set(np.unique(lab).astype(int).tolist()) <= set(range(n_cls + 1))
              and (lab > 0).any(), str(np.unique(lab).astype(int).tolist()))
        check(f"data_ocp max {int(ocp.max())} == particle count {len(co)}",
              int(ocp.max()) == len(co))

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
