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
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

NUM_CLASSES = 6

# Tomogram id = patch filename minus the trailing _<x>_<y>_<z>.pt offsets.
PATCH_RE = re.compile(r'^(?P<tomo>.+?)_(?P<x>\d+)_(?P<y>\d+)_(?P<z>\d+)\.pt$')


def parse_patch_name(image_path):
    """'.../TS_5_4_0_64_56.pt' -> ('TS_5_4', 0, 64, 56)."""
    m = PATCH_RE.match(os.path.basename(image_path))
    if m is None:
        raise ValueError(f"Cannot parse patch filename: {image_path}")
    return m.group('tomo'), int(m.group('x')), int(m.group('y')), int(m.group('z'))


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
    args = ap.parse_args()

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
        tomos_here = {parse_patch_name(e['image'])[0] for e in items}
        others = set().union(*[{parse_patch_name(e['image'])[0] for e in v}
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
    }

    with open(args.output_json, 'w') as f:
        json.dump(split, f)

    # --- summary --------------------------------------------------------------------
    print(f"wrote {args.output_json}")
    print(f"{'split':<12}{'tomograms':<34}{'patches':>9}{'GT':>7}   per-class 0..5")
    for name in ('training', 'validation', 'test'):
        items = split[name]
        tomos = sorted({parse_patch_name(e['image'])[0] for e in items})
        counts = Counter(int(p[3]) for e in items for p in e['points'] if p[3] >= 0)
        print(f"{name:<12}{','.join(tomos):<34}{len(items):>9}"
              f"{sum(counts.values()):>7}   {[counts.get(i, 0) for i in range(NUM_CLASSES)]}")
    total = sum(len(split[s]) for s in ('training', 'validation', 'test'))
    print(f"\ntotal patches {total} (source had {len(entries)})"
          + ("" if args.deoverlap_eval else "; no patches dropped"))
    if args.deoverlap_eval:
        print("NOTE --deoverlap-eval dropped overlapping eval patches; rare-class counts above "
              "may be too small for a stable per-class F4.")


if __name__ == '__main__':
    main()
