#!/usr/bin/env python
"""Write DeepETPicker's preprocessing config from the dataset_split.json prepare_from_datalist.py
produced, and print the train/val dataset ids for generate_train_config.py.

The config format is the one bin/preprocess.py parses: a file whose contents are the literal
`pre_config=` followed by a JSON object (configs/SHREC_2021_preprocess.py is the upstream sample).
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="base_path (holds raw_data/ and dataset_split.json)")
    ap.add_argument("--dset-name", default="Dataset440_preprocess")
    ap.add_argument("--label-diameter", type=int, default=9,
                    help="diameter of the weak gaussian label; must be smaller than the smallest "
                         "real particle diameter (default 9)")
    ap.add_argument("--norm-type", default="standardization",
                    choices=["standardization", "normalization"])
    ap.add_argument("--val-id", type=int, default=None,
                    help="dataset id held out for validation (default: second to last)")
    ap.add_argument("--test-id", type=int, default=None,
                    help="dataset id held out for the reported metrics (default: last)")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    with open(os.path.join(out, "dataset_split.json")) as f:
        split = json.load(f)

    n = len(split["tomograms"])
    val_id = split.get("val_id", n - 2) if args.val_id is None else args.val_id
    test_id = split.get("test_id", n - 1) if args.test_id is None else args.test_id
    if not (0 <= val_id < n and 0 <= test_id < n):
        raise SystemExit(f"val/test id out of range for {n} tomograms")

    diams = [int(d) for d in split["ocp_diameter"].split(",")]
    smallest = min(diams)
    if args.label_diameter >= smallest:
        raise SystemExit(f"--label-diameter {args.label_diameter} must be < smallest particle "
                         f"diameter {smallest}")
    # utils/coords2labels.py pastes a `dim`-edge template into a 2*(dim//2)+1 window, so an even
    # diameter is off by one and the label/occupancy pass dies mid-multiprocessing.
    even = [d for d in diams + [args.label_diameter] if d % 2 == 0]
    if even:
        raise SystemExit(f"diameters must be odd, got even: {even} -- DeepETPicker's "
                         f"coords2labels.py cannot broadcast an even-edge template")

    pre_config = {
        "dset_name": args.dset_name,
        "base_path": out,
        "coord_path": os.path.join(out, "raw_data"),
        "coord_format": ".coords",
        "tomo_path": os.path.join(out, "raw_data"),
        "tomo_format": ".mrc",
        "num_cls": split["num_cls"],
        "label_type": "gaussian",
        "label_diameter": args.label_diameter,   # int: the label folder is named gaussian<N>
        "ocp_type": "sphere",
        "ocp_diameter": split["ocp_diameter"],
        "norm_type": args.norm_type,
    }
    cfg_path = os.path.join(out, f"{args.dset_name}.py")
    with open(cfg_path, "w") as f:
        f.write("pre_config=")
        json.dump(pre_config, f, indent=12)

    # train ids must be the contiguous block 0..n-1: bin/train_bash.py derives data_split from
    # len(train_list), not from the ids themselves.
    train_ids = [i for i in range(n) if i not in (val_id, test_id)]
    if train_ids != list(range(len(train_ids))):
        raise SystemExit(f"train ids {train_ids} are not contiguous from 0 -- pick val/test ids "
                         f"at the end of the sorted tomogram list {split['tomograms']}")
    train_arg = f"{train_ids[0]}-{train_ids[-1]}" if len(train_ids) > 1 else str(train_ids[0])

    split.update({"val_id": val_id, "test_id": test_id, "train_ids": train_ids,
                  "label_diameter": args.label_diameter})
    with open(os.path.join(out, "dataset_split.json"), "w") as f:
        json.dump(split, f, indent=2)

    names = split["tomograms"]
    print(f"pre_config -> {cfg_path}")
    print(f"  num_cls={pre_config['num_cls']}  ocp_diameter={pre_config['ocp_diameter']}  "
          f"label_diameter={args.label_diameter}")
    print(f"  train ids {train_arg}: {[names[i] for i in train_ids]}")
    print(f"  val   id  {val_id}: {names[val_id]}")
    print(f"  test  id  {test_id}: {names[test_id]}")
    # consumed by run_datalist.sh
    print(f"TRAIN_SET_IDS={train_arg}")
    print(f"VAL_SET_IDS={val_id}")
    print(f"TEST_TOMO={names[test_id]}")
    print(f"VAL_TOMO={names[val_id]}")
    print(f"LABEL_NAME=gaussian{args.label_diameter}")


if __name__ == "__main__":
    main()
