"""
Build an honest train/val/test split for the CZII detection patches.

Why this exists
---------------
The shipped ``czi_100_datalist.json`` cannot be used for evaluation: every one of its
121 ``validation`` and 364 ``test`` entries contains a single padding row
``[-100, -100, -100, -100, -100]``, i.e. ZERO ground truth. ``CZIDetectionMetrics``
filters GT on ``points[:, 3] >= 0``, so recall is 0 and F4 is identically 0.0 at every
threshold — which also makes ``best_model.pth`` selection meaningless. Only the
``training`` split carries labels.

The obvious workaround — hold out some training patches — leaks badly. Patches are cut
on a sliding window with stride = patch_size // 2, so neighbours OVERLAP 50% in x/y
(and ~56% in z) and share the same particle instances:

    x, y offsets: 0, 64, 128, ..., 448, 502   (patch size 128)
    z offsets:    0, 56
    => 9 x 9 x 2 = 162 patches per tomogram

A random patch-level split therefore puts half of a validation patch's volume into the
training set. This script splits by TOMOGRAM instead, which removes the leakage.

Split
-----
5 tomograms train / 1 val / 1 test (there are only 7). Defaults pick val and test with
near-median GT counts, avoiding the extremes:

    TS_5_4   871 GT      TS_73_6  1157 GT  <- default test
    TS_69_2 1009 GT  <- default val
    TS_6_4  1195 GT      TS_86_3  1370 GT
    TS_6_6   872 GT      TS_99_9  1226 GT

Class IDs are NOT touched: they stay exactly as ``load_detection_annotations()`` in
downstream_patch_generation.py assigned them (alphabetical, 0-indexed), which is also
what metrics.py assumes.

Usage
-----
    python make_detection_split.py \\
        --datalist-json /path/to/Dataset440_.../czi_100_datalist.json \\
        --images-dir    /path/to/Dataset440_.../images \\
        --output-json   /path/to/czi_split_datalist.json

    # hold out only non-overlapping eval patches (see --deoverlap-eval caveat below)
    python make_detection_split.py ... --deoverlap-eval

    # val/test as FULL tomograms (train stays .pt patches); detection3d.py then runs
    # sliding-window inference over each whole tomogram. Feed the output via --datalist-json.
    python make_detection_split.py ... \\
        --tomo-dir /path/to/Dataset440_CZII_10440/imagesTr \\
        --csv      /path/to/Dataset440_CZII_10440/point_annotations.csv
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

NUM_CLASSES = 6

# Same radii (Å) as slurm_scripts/patchify_detection_czi.sh, so full-tomogram GT sigmas match
# the sigmas baked into the training patches.
CZI_SIGMAS_ANG = ('{"Beta-amylase":65,"Beta-galactosidase":90,"Thyroglobulin":130,'
                  '"cytosolic ribosome":150,"ferritin complex":60,"virus-like capsid":135}')

# Tomogram id = patch filename minus the trailing _<x>_<y>_<z>.pt offsets.
PATCH_RE = re.compile(r'^(?P<tomo>.+?)_(?P<x>\d+)_(?P<y>\d+)_(?P<z>\d+)\.pt$')


def parse_patch_name(image_path):
    """'.../TS_5_4_0_64_56.pt' -> ('TS_5_4', 0, 64, 56)."""
    m = PATCH_RE.match(os.path.basename(image_path))
    if m is None:
        raise ValueError(f"Cannot parse patch filename: {image_path}")
    return m.group('tomo'), int(m.group('x')), int(m.group('y')), int(m.group('z'))


def tomo_of(image_path):
    """Tomogram id of a datalist entry: a .pt patch or a full <run>_0000.nii.gz tomogram."""
    base = os.path.basename(image_path)
    if base.endswith('.nii.gz'):
        return re.sub(r'_\d{4}\.nii\.gz$', '', base)
    return parse_patch_name(image_path)[0]


def n_valid_points(entry):
    """GT rows only — padding rows carry class_id == -100 in column 3."""
    return sum(1 for p in entry['points'] if p[3] >= 0)


def is_non_overlapping(x, y, z, patch_size):
    """Keep only patches whose offsets land on a stride-`patch_size` grid.

    NOTE this is aggressive: on this dataset it keeps x, y in {0, 128, 256, 384} and
    z == 0, i.e. 16 of 162 patches (~10%), and drops coverage of the z 56-184 slab and
    the clamped x/y 502 edge. See --deoverlap-eval help for why it is off by default.
    """
    return x % patch_size == 0 and y % patch_size == 0 and z % patch_size == 0


def full_tomo_entry(tomo, tomo_dir, annotations):
    """One val/test entry for a whole tomogram: nii.gz path + global XYZ GT points."""
    nii_path = os.path.join(tomo_dir, f"{tomo}_0000.nii.gz")
    if not os.path.isfile(nii_path):
        raise SystemExit(f"Tomogram not found: {nii_path}")
    if tomo not in annotations:
        raise SystemExit(f"No annotations for {tomo!r} in --csv")
    info = annotations[tomo]
    return {'image': nii_path, 'points': info['points'], 'voxel_size': info['voxel_size']}


def check_patch_points_match_csv(by_tomo, annotations, tol=1e-3):
    """Every training-patch point + its patch offset must be a CSV (global) point of that run.

    Catches axis-order / offset mismatches between the patches and the full-tomogram GT,
    which would otherwise silently make val/test F4 meaningless.
    """
    import numpy as np
    for tomo, entries in by_tomo.items():
        gt = np.asarray(annotations[tomo]['points'], dtype=np.float64)
        for e in entries:
            x0, y0, z0 = e['_offsets']
            for p in e['points']:
                if p[3] < 0:
                    continue
                g = np.array([p[0] + x0, p[1] + y0, p[2] + z0, p[3], p[4]])
                if not np.any(np.all(np.abs(gt - g) < tol, axis=1)):
                    raise AssertionError(
                        f"patch point {p} in {e['image']} (offset {x0},{y0},{z0}) has no "
                        f"matching CSV point in {tomo}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datalist-json', required=True,
                    help="Source datalist; only its 'training' split is read (the only one with GT).")
    ap.add_argument('--output-json', required=True, help="Where to write the new datalist.")
    ap.add_argument('--images-dir', default=None,
                    help="Rewrite every image path to this directory (the shipped paths are "
                         "cluster-absolute and will not resolve elsewhere). Default: keep as-is.")
    ap.add_argument('--val-tomo', default='TS_69_2', help="Tomogram id held out for validation.")
    ap.add_argument('--test-tomo', default='TS_73_6', help="Tomogram id held out for test.")
    ap.add_argument('--patch-size', type=int, default=128)
    ap.add_argument('--deoverlap-eval', action='store_true',
                    help="Keep only non-overlapping patches in val/test. OFF by default: splitting "
                         "by tomogram ALREADY removes all train/eval leakage, so the residual "
                         "overlap merely re-weights particles in overlap zones rather than "
                         "inflating the score. Turning this on discards ~90%% of the eval data and "
                         "leaves the rarest class at ~6 instances, where one detection swings F4 "
                         "by ~0.17.")
    ap.add_argument('--tomo-dir', default=None,
                    help="Directory with full tomograms <run>_0000.nii.gz. With --csv, val/test "
                         "become one full-tomogram entry each (global GT) instead of patches.")
    ap.add_argument('--csv', default=None,
                    help="Point annotations CSV (run,particle_name,z,y,x,voxel_size) for full-tomogram val/test GT.")
    ap.add_argument('--sigmas-ang', default=CZI_SIGMAS_ANG,
                    help="JSON {particle_name: radius_Å} for GT sigmas (default: CZI radii).")
    args = ap.parse_args()
    full_tomo = bool(args.tomo_dir or args.csv)
    if full_tomo and not (args.tomo_dir and args.csv):
        ap.error("--tomo-dir and --csv must be given together")
    if full_tomo and args.deoverlap_eval:
        ap.error("--deoverlap-eval only applies to patch-based val/test, not --tomo-dir")

    with open(args.datalist_json) as f:
        source = json.load(f)
    entries = source['training']
    if not entries:
        raise SystemExit("Source 'training' split is empty — nothing to split.")

    by_tomo = defaultdict(list)
    for e in entries:
        tomo, x, y, z = parse_patch_name(e['image'])
        entry = dict(e)
        if args.images_dir:
            entry['image'] = os.path.join(args.images_dir, os.path.basename(e['image']))
        entry['_offsets'] = (x, y, z)
        by_tomo[tomo].append(entry)

    tomograms = sorted(by_tomo)
    for name, tomo in (('--val-tomo', args.val_tomo), ('--test-tomo', args.test_tomo)):
        if tomo not in by_tomo:
            raise SystemExit(f"{name}={tomo!r} not found. Available: {', '.join(tomograms)}")
    if args.val_tomo == args.test_tomo:
        raise SystemExit("--val-tomo and --test-tomo must differ.")

    train_tomos = [t for t in tomograms if t not in (args.val_tomo, args.test_tomo)]
    if not train_tomos:
        raise SystemExit("No tomograms left for training.")

    def collect(tomos, deoverlap):
        out = []
        for t in tomos:
            for e in by_tomo[t]:
                x, y, z = e['_offsets']
                if deoverlap and not is_non_overlapping(x, y, z, args.patch_size):
                    continue
                out.append({k: v for k, v in e.items() if k != '_offsets'})
        return out

    if full_tomo:
        from downstream_patch_generation import load_detection_annotations
        annotations, _ = load_detection_annotations(args.csv, json.loads(args.sigmas_ang), 0.0)
        check_patch_points_match_csv(by_tomo, annotations)
        split = {
            'training':   collect(train_tomos, deoverlap=False),
            'validation': [full_tomo_entry(args.val_tomo, args.tomo_dir, annotations)],
            'test':       [full_tomo_entry(args.test_tomo, args.tomo_dir, annotations)],
        }
    else:
        split = {
            'training':   collect(train_tomos, deoverlap=False),
            'validation': collect([args.val_tomo], deoverlap=args.deoverlap_eval),
            'test':       collect([args.test_tomo], deoverlap=args.deoverlap_eval),
        }

    # --- assertions: the entire point of this script -------------------------------
    assignment = {t: 'training' for t in train_tomos}
    assignment[args.val_tomo] = 'validation'
    assignment[args.test_tomo] = 'test'
    assert len(assignment) == len(tomograms), "a tomogram was assigned to more than one split"

    for name, items in split.items():
        assert items, f"split {name!r} is empty"
        tomos_here = {tomo_of(e['image']) for e in items}
        others = set().union(*[{tomo_of(e['image']) for e in v}
                               for k, v in split.items() if k != name])
        assert not (tomos_here & others), \
            f"LEAKAGE: {sorted(tomos_here & others)} appears in {name!r} and another split"
        assert sum(n_valid_points(e) for e in items) > 0, f"split {name!r} has no ground truth"
        for e in items:
            for p in e['points']:
                assert p[3] < 0 or 0 <= int(p[3]) < NUM_CLASSES, \
                    f"class_id {p[3]} out of range in {e['image']}"

    split['_provenance'] = {
        'source': os.path.abspath(args.datalist_json),
        'split_by': 'tomogram (patches overlap 50%, so patch-level splits leak)',
        'train_tomograms': train_tomos,
        'val_tomogram': args.val_tomo,
        'test_tomogram': args.test_tomo,
        'deoverlap_eval': args.deoverlap_eval,
        'eval_mode': 'full_tomogram' if full_tomo else 'patches',
    }

    with open(args.output_json, 'w') as f:
        json.dump(split, f)

    # --- summary --------------------------------------------------------------------
    print(f"wrote {args.output_json}")
    print(f"{'split':<12}{'tomograms':<34}{'entries':>9}{'GT':>7}   per-class 0..5")
    for name in ('training', 'validation', 'test'):
        items = split[name]
        tomos = sorted({tomo_of(e['image']) for e in items})
        counts = Counter(int(p[3]) for e in items for p in e['points'] if p[3] >= 0)
        print(f"{name:<12}{','.join(tomos):<34}{len(items):>9}"
              f"{sum(counts.values()):>7}   {[counts.get(i, 0) for i in range(NUM_CLASSES)]}")
    if full_tomo:
        print("\nval/test are FULL tomograms (global GT); training patch points verified "
              "against --csv. Pass this json to detection3d.py via --datalist-json.")
    else:
        total = sum(len(split[s]) for s in ('training', 'validation', 'test'))
        print(f"\ntotal patches {total} (source had {len(entries)})"
              + ("" if args.deoverlap_eval else "; no patches dropped"))
    if args.deoverlap_eval:
        print("NOTE --deoverlap-eval dropped overlapping eval patches; rare-class counts above "
              "may be too small for a stable per-class F4.")


if __name__ == '__main__':
    main()
