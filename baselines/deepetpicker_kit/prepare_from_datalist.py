#!/usr/bin/env python
"""Build a DeepETPicker base_path from a detection datalist JSON (Dataset440 style).

DeepETPicker works on whole tomograms: a raw_data folder holding <tomo>.mrc next to
<tomo>.coords. The datalist instead gives 128^3 patches with point labels, so this script
reassembles each tomogram from its patches and merges the points.

Reassembly is exact: the patches are crops of one globally normalised volume, so overlapping
patches agree bit-for-bit (asserted below).

Axis convention -- the one thing to get right:
    patch name TS_a_b_<i>_<j>_<k>, point [p0, p1, p2, cls, radius]
    global voxel  = (p0+i, p1+j, p2+k)   and the assembled array is V[p0+i, p1+j, p2+k]
    MRC wants (z, y, x), and k is the short axis (184), so we write V.transpose(2, 1, 0)
    .coords rows are  "cls+1 <tab> i <tab> j <tab> k"  because utils/coords2labels.py reads
    `cls, x, y, z = row` and then indexes `label_data[z, y, x]` on that (z, y, x) volume.

Writes:
  <out>/raw_data/<tomo>.mrc       float32, (184, 630, 630) for Dataset440
  <out>/raw_data/<tomo>.coords    tab-separated ints, cls in 1..num_cls
  <out>/dataset_split.json        tomogram order (== DeepETPicker dataset ids), counts, radii
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

from datalist_utils import load_datalist, load_patch, parse_patch_name, save_mrc, strip_ext

PATCH = 128  # cube edge of the datalist patches

# Dataset440 ground truth, used as a self-check (skip with --no-expect-check).
EXPECTED_PARTICLES = {
    "TS_5_4": 140, "TS_69_2": 143, "TS_6_4": 193, "TS_6_6": 143,
    "TS_73_6": 216, "TS_86_3": 224, "TS_99_9": 205,
}


def odd_up(x):
    """Round a diameter up to the nearest odd integer (see the comment in main())."""
    d = int(np.ceil(x))
    return d + 1 if d % 2 == 0 else d


def group_by_tomogram(entries):
    """[{image, points}] -> {tomo: [(offset, points), ...]} , skipping non-patch entries."""
    groups = collections.defaultdict(list)
    skipped = []
    for e in entries:
        name = strip_ext(e["image"])
        try:
            tomo, off = parse_patch_name(name)
        except ValueError:
            skipped.append(name)
            continue
        groups[tomo].append((off, e.get("points", []), e["image"]))
    return groups, skipped


def assemble(patches, image_root):
    """Paste the patches into one array; assert overlaps agree and everything is covered."""
    shape = tuple(max(off[d] for off, _, _ in patches) + PATCH for d in range(3))
    vol = np.zeros(shape, dtype=np.float32)
    seen = np.zeros(shape, dtype=bool)
    for off, _, path in patches:
        cube = load_patch(path, image_root)
        if cube.shape != (PATCH,) * 3:
            raise SystemExit(f"unexpected patch shape {cube.shape} for {path}")
        sl = tuple(slice(off[d], off[d] + PATCH) for d in range(3))
        overlap = seen[sl]
        if overlap.any():
            prev = vol[sl][overlap]
            if not np.array_equal(prev, cube[overlap]):
                bad = np.abs(prev - cube[overlap]).max()
                raise SystemExit(f"patches disagree on their overlap (max|diff|={bad}) at {path}")
        vol[sl] = cube
        seen[sl] = True
    if not seen.all():
        raise SystemExit(f"assembled volume has {int((~seen).sum())} uncovered voxels")
    return vol


def merge_points(patches):
    """Patch-local points -> unique global (x, y, z, cls) rows, plus {cls: radius}."""
    uniq, radii = {}, {}
    for off, points, _ in patches:
        for p in points:
            cls = int(p[3])
            if cls < 0:                      # -100 = "this patch has no annotation"
                continue
            key = tuple(round(p[d] + off[d], 2) for d in range(3)) + (cls,)
            uniq[key] = (p[0] + off[0], p[1] + off[1], p[2] + off[2], cls)
            radii.setdefault(cls, float(p[4]))
    rows = sorted(uniq.values(), key=lambda r: (r[3], r[0], r[1], r[2]))
    return rows, radii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", required=True)
    ap.add_argument("--out", required=True, help="DeepETPicker base_path to create")
    ap.add_argument("--image-root", default=None,
                    help="local dir holding the .pt patches (default: <datalist dir>/images); "
                         "the datalist records the paths of the machine it was built on")
    ap.add_argument("--split", default="training",
                    help="which datalist split holds the labelled patches (default: training)")
    ap.add_argument("--no-expect-check", action="store_true",
                    help="skip the Dataset440 particle-count self-check")
    ap.add_argument("--force", action="store_true",
                    help="re-assemble tomograms whose .mrc already exists (reassembly reads ~9 GB "
                         "of patches, so by default an existing .mrc is reused)")
    ap.add_argument("--dry-run", action="store_true", help="report the grouping and stop")
    args = ap.parse_args()

    image_root = args.image_root or os.path.join(os.path.dirname(os.path.abspath(args.datalist)),
                                                 "images")
    d = load_datalist(args.datalist)
    print(f"datalist: " + " ".join(f"{k}={len(d[k])}" for k in ("training", "validation", "test")))

    groups, skipped = group_by_tomogram(d[args.split])
    if skipped:
        print(f"note: {len(skipped)} '{args.split}' entries are not 128^3 patches and were skipped "
              f"(e.g. {skipped[0]}) -- whole-tomogram entries need their own volumes on disk")
    if not groups:
        raise SystemExit(f"no usable patches in split '{args.split}'")
    tomos = sorted(groups)   # == the order coords_gen/num_name.csv will assign dataset ids in
    print(f"{len(tomos)} tomograms from split '{args.split}': " +
          ", ".join(f"{t}({len(groups[t])}p)" for t in tomos))

    if args.dry_run:
        return

    raw = os.path.join(args.out, "raw_data")
    os.makedirs(raw, exist_ok=True)

    all_radii, counts, shapes = {}, {}, {}
    for idx, tomo in enumerate(tomos):
        patches = groups[tomo]
        rows, radii = merge_points(patches)
        for c, r in radii.items():
            all_radii.setdefault(c, r)
            if abs(all_radii[c] - r) > 1e-3:
                raise SystemExit(f"class {c} has two radii: {all_radii[c]} vs {r}")

        mrc_path = os.path.join(raw, tomo + ".mrc")
        if os.path.exists(mrc_path) and not args.force:
            import mrcfile
            with mrcfile.open(mrc_path, permissive=True, header_only=True) as m:
                zyx = (int(m.header.nz), int(m.header.ny), int(m.header.nx))
            shape = zyx[::-1]
            reused = " (reused existing mrc)"
        else:
            vol = assemble(patches, image_root)
            save_mrc(vol.transpose(2, 1, 0), mrc_path)
            shape = vol.shape
            reused = ""

        coords = np.array([[r[3] + 1, r[0], r[1], r[2]] for r in rows])
        np.savetxt(os.path.join(raw, tomo + ".coords"), np.rint(coords).astype(int),
                   fmt="%d", delimiter="\t", newline="\n")

        counts[tomo], shapes[tomo] = len(rows), list(shape)
        per_cls = collections.Counter(r[3] for r in rows)
        print(f"  [id {idx}] {tomo:10s} vol={tuple(shape)} -> mrc{tuple(shape[::-1])}  "
              f"particles={len(rows):4d}  per-class={dict(sorted(per_cls.items()))}{reused}",
              flush=True)

        if not args.no_expect_check and tomo in EXPECTED_PARTICLES:
            exp = EXPECTED_PARTICLES[tomo]
            if len(rows) != exp:
                raise SystemExit(f"{tomo}: expected {exp} unique particles, got {len(rows)}")

    classes = sorted(all_radii)
    # Diameters must be ODD: utils/coords2labels.py builds a template of edge `dim` but pastes it
    # into a window of edge 2*(dim//2)+1, which for even dim is one voxel larger ("operands could
    # not be broadcast together with shapes (19,19,19) (18,18,18)").  Upstream only ever used odd
    # diameters (SHREC 7..17, EMPIAR 23).  Round up so the occupancy sphere is never too small.
    diameters = [odd_up(2 * all_radii[c]) for c in classes]
    split = {
        "tomograms": tomos,                      # index == DeepETPicker dataset id
        "particle_counts": counts,
        "volume_shape_xyz": shapes,
        "classes": classes,                      # 0-based datalist class ids
        "radii_voxels": {str(c): all_radii[c] for c in classes},
        "diameters_voxels": {str(c): diameters[i] for i, c in enumerate(classes)},
        "ocp_diameter": ",".join(str(x) for x in diameters),
        "num_cls": len(classes),
        "raw_data": raw,
    }
    with open(os.path.join(args.out, "dataset_split.json"), "w") as f:
        json.dump(split, f, indent=2)

    print(f"\ntotal particles: {sum(counts.values())}")
    print(f"classes {classes} -> ocp_diameter '{split['ocp_diameter']}' (voxels)")
    print(f"raw_data ready: {raw}\nsplit written:  {os.path.join(args.out, 'dataset_split.json')}")


if __name__ == "__main__":
    sys.exit(main())
