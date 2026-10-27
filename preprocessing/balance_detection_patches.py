"""Balance a detection datalist's training patches 1:1 foreground : background.

Keeps every foreground patch (>= 1 labelled point, class_id >= 0) and draws the same number of
background patches at random (seeded). validation / test are copied unchanged: they are whole
tomograms run through sliding-window inference, so no background selection applies there.

The exhaustive datalist is kept next to it as <name>_fullpatches.json and is always the input,
so re-running gives the same balanced file.

    python preprocessing/balance_detection_patches.py \
        --datalist .../byu_detection_patches_128/byu_100_datalist.json \
        .../byu_detection_patches_128/byu_50percent/byu_100_datalist.json \
        .../byu_detection_patches_128/byu_10percent/byu_100_datalist.json
"""
import argparse
import json
import os
import random
import shutil


def is_foreground(entry):
    return any(p[3] >= 0 for p in entry["points"])


def balance(datalist_path, seed):
    full_path = datalist_path[:-len(".json")] + "_fullpatches.json"
    if not os.path.exists(full_path):
        shutil.copy2(datalist_path, full_path)
    with open(full_path) as f:
        d = json.load(f)

    fg = [i for i, e in enumerate(d["training"]) if is_foreground(e)]
    bg = [i for i, e in enumerate(d["training"]) if not is_foreground(e)]
    if len(bg) < len(fg):
        raise SystemExit(f"{full_path}: fewer background ({len(bg)}) than foreground ({len(fg)}) patches")
    keep = sorted(fg + random.Random(seed).sample(bg, len(fg)))

    out = dict(d, training=[d["training"][i] for i in keep])
    tmp = datalist_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, datalist_path)
    print(f"{datalist_path}: training {len(d['training'])} -> {len(keep)} "
          f"({len(fg)} fg + {len(fg)} bg, seed {seed}); "
          f"val {len(d['validation'])} / test {len(d['test'])} unchanged; full list: {full_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for p in args.datalist:
        balance(p, args.seed)


if __name__ == "__main__":
    main()
