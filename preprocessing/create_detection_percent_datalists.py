"""Create tomogram-level 10% / 50% training subsets of a detection patch datalist (BYU, CZI).

Mirrors the segmentation *_10percent / *_50percent datalists:
  - whole tomograms only (a tomogram's patches are all in or all out)
  - n_tomos = ceil(pct * N), random tomograms (seeded)
  - nested: the 10% tomograms are a subset of the 50% tomograms
  - validation / test are copied unchanged

If the split datalist entries carry "num_motors" (BYU, create_byu_detection_split.py),
the pick is done per motor-count bin (0 / 1 / 2+) so the subset keeps the full split's
balance. Otherwise (CZI) tomograms are picked uniformly at random.

detection3d / loaders.py only accept dataset_name in {'byu', 'czi'} and read
"{base-data-dir}/{dataset_name}_100_datalist.json", so each subset gets its own folder:

  {patch-dir}/{name}_10percent/{name}_100_datalist.json
  {patch-dir}/{name}_50percent/{name}_100_datalist.json

Patch paths inside stay absolute, so the .pt files are shared, not copied.
Train with --dataset-name {name} --base-data-dir {patch-dir}/{name}_10percent.

Usage:
    python preprocessing/create_detection_percent_datalists.py \\
        --dataset-name byu \\
        --split-datalist .../byu_dataset/byu_split_datalist.json \\
        --patch-datalist .../byu_detection_patches_128/byu_100_datalist.json \\
        --percents 10 50 --seed 42
"""

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


def run_name(nii_path: str) -> str:
    """'TS_5_4_0000.nii.gz' → 'TS_5_4' (same rule as downstream_patch_generation.run_name_from_nii)."""
    base = os.path.basename(nii_path).replace(".nii.gz", "").replace(".nii", "")
    if "_" in base and base[-5] == "_" and base[-4:].isdigit():
        base = base[:-5]
    return base


def patch_tomo(patch_path: str) -> str:
    """'.../TS_5_4_0_64_128.pt' → 'TS_5_4'  (downstream_patch_generation: {run}_{x0}_{y0}_{z0}.pt)."""
    return os.path.basename(patch_path)[:-len(".pt")].rsplit("_", 3)[0]


def strata(entry: dict) -> str:
    """Motor-count bin (0 / 1 / 2+) when available, else a single stratum."""
    if "num_motors" not in entry:
        return "all"
    return str(entry["num_motors"]) if entry["num_motors"] < 2 else "2+"


def nested_subsets(tomo_strata: dict, percents: list, seed: int) -> dict:
    """Shuffle each stratum once, then take ceil(pct * n_stratum) from the front → nested subsets."""
    by_stratum = defaultdict(list)
    for tomo_id in sorted(tomo_strata):
        by_stratum[tomo_strata[tomo_id]].append(tomo_id)

    rng = random.Random(seed)
    for name in sorted(by_stratum):
        rng.shuffle(by_stratum[name])

    return {
        pct: {t for ids in by_stratum.values() for t in ids[:math.ceil(len(ids) * pct / 100)]}
        for pct in percents
    }


def main():
    parser = argparse.ArgumentParser(description="Tomogram-level percent subsets of a detection patch datalist.")
    parser.add_argument("--dataset-name", required=True, help="byu or czi (names the output folder/file)")
    parser.add_argument("--split-datalist", type=Path, required=True, help="Tomogram split JSON fed to patch generation")
    parser.add_argument("--patch-datalist", type=Path, required=True, help="{name}_100_datalist.json from patch generation")
    parser.add_argument("--percents", type=int, nargs="+", default=[10, 50], help="Subset sizes in %% (default: 10 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    split = json.loads(args.split_datalist.read_text())
    tomo_strata = {run_name(e["image"]): strata(e) for e in split["training"]}

    full = json.loads(args.patch_datalist.read_text())
    unknown = {patch_tomo(e["image"]) for e in full["training"]} - set(tomo_strata)
    if unknown:
        raise ValueError(f"{len(unknown)} patch tomograms not in split training list, e.g. {sorted(unknown)[:3]}")

    subsets = nested_subsets(tomo_strata, args.percents, args.seed)

    n_tomos, n_patches = len(tomo_strata), len(full["training"])
    print(f"100%: {n_tomos} tomos  {n_patches} patches")
    for pct in sorted(subsets):
        tomos = subsets[pct]
        train = [e for e in full["training"] if patch_tomo(e["image"]) in tomos]
        counts = Counter(tomo_strata[t] for t in tomos)

        out_dir = args.patch_datalist.parent / f"{args.dataset_name}_{pct}percent"
        out_path = out_dir / f"{args.dataset_name}_100_datalist.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            {"training": train, "validation": full["validation"], "test": full["test"]}, indent=2))

        strata_str = "  ".join(f"{k}:{counts[k]}" for k in sorted(counts))
        print(f"{pct:>3}%: {len(tomos)} tomos ({100 * len(tomos) / n_tomos:.1f}%)  "
              f"{len(train)} patches ({100 * len(train) / n_patches:.1f}%)  [{strata_str}]  → {out_path}")
        if len(tomos) <= 15:
            print(f"      tomos: {sorted(tomos)}")

    pcts = sorted(subsets)
    for small, large in zip(pcts, pcts[1:]):
        assert subsets[small] <= subsets[large], f"{small}% not nested in {large}%"


if __name__ == "__main__":
    main()
