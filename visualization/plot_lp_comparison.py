"""
Aggregate linear-probing results.json files and plot B200-vs-H100 comparison.

Reads every run dir under --base-dir, parses (backbone family, checkpoint
iteration, dataset) from its name, pulls test_dice from results.json, and
plots per-dataset: B200 Dice vs. pretraining iteration for each stage
(pretrain/highres112/highres128), with H100-best and random-init as flat
reference lines. Also writes a CSV summary.

Usage (run on the cluster login node — no GPU needed):
    python visualization/plot_lp_comparison.py \
        --base-dir /cluster/projects/bwanggroup/reza/projects/cryoet/experiments/linear_probing_h100_b200_comparison \
        --out-dir  /cluster/projects/bwanggroup/reza/projects/cryoet/experiments/linear_probing_h100_b200_comparison/plots
"""

import argparse
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = [
    "Dataset001_CZII_10001_patches512",
    "Dataset010_CZII_10010_patches512",
    "Dataset989_EMPIAR_10989_transposed_patches512",
    "Dataset049_EMPIAR_12049_transposed_patches512",
]
DATASET_SHORT = {
    "Dataset001_CZII_10001_patches512": "CZII-10001",
    "Dataset010_CZII_10010_patches512": "CZII-10010",
    "Dataset989_EMPIAR_10989_transposed_patches512": "EMPIAR-10989",
    "Dataset049_EMPIAR_12049_transposed_patches512": "EMPIAR-12049",
}

# (family label, regex to pull iteration, plot color)
FAMILY_RE = {
    "b200_pretrain": re.compile(r"^b200_pretrain_training_(\d+)_"),
    "b200_highres112": re.compile(r"^b200_highres112_training_(\d+)_"),
    "b200_highres128": re.compile(r"^b200_highres128_training_(\d+)_"),
}
FAMILY_COLOR = {
    "b200_pretrain": "#4DBBD5",
    "b200_highres112": "#00A087",
    "b200_highres128": "#3C5488",
}
FAMILY_LABEL = {
    "b200_pretrain": "B200 pretrain (96³)",
    "b200_highres112": "B200 high-res (112³)",
    "b200_highres128": "B200 high-res (128³)",
}


def parse_dirname(name):
    for ds in DATASETS:
        if name.endswith(ds):
            prefix = name[: -len(ds) - 1]  # strip "_<dataset>"
            for family, rgx in FAMILY_RE.items():
                m = rgx.match(prefix + "_")
                if m:
                    return family, int(m.group(1)), ds
            if prefix == "h100_highres_training_9374":
                return "h100_best", 9374, ds
            if prefix == "random_init_randominit":
                return "random_init", None, ds
    return None, None, None


def load_results(base_dir):
    rows = []
    for d in sorted(glob.glob(os.path.join(base_dir, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        family, it, ds = parse_dirname(name)
        if family is None:
            continue
        rj = os.path.join(d, "results.json")
        bm = os.path.join(d, "best_model.pth")
        if not (os.path.isfile(rj) and os.path.isfile(bm)):
            continue
        with open(rj) as f:
            r = json.load(f)
        rows.append({
            "family": family, "iteration": it, "dataset": ds,
            "test_dice": r.get("test_dice"),
            "test_per_cls_dice": r.get("test_per_cls_dice"),
        })
    return rows


def write_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "iteration", "dataset", "test_dice", "test_per_cls_dice"])
        for r in rows:
            w.writerow([r["family"], r["iteration"], r["dataset"], r["test_dice"], r["test_per_cls_dice"]])
    print(f"Saved CSV: {out_path} ({len(rows)} rows)")


# Cumulative offset: high-res adaptation runs AFTER pretraining finishes, so its
# checkpoint iterations (0-based within that stage) must be shifted onto one
# continuous training timeline instead of overlaid on pretrain's own 0-124999
# range. highres112 and highres128 are TWO SEPARATE adaptation experiments that
# both branch off the SAME final pretrain checkpoint (training_112499) — they
# are parallel alternatives, not sequential — so they share the same offset.
PRETRAIN_TOTAL_ITERS = 125000       # b200_pretrain: 0-124999
OFFSET = {
    "b200_pretrain": 0,
    "b200_highres112": PRETRAIN_TOTAL_ITERS,
    "b200_highres128": PRETRAIN_TOTAL_ITERS,
}


def plot_dataset(rows, ds, out_path, dpi=300):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "axes.linewidth": 1.0, "axes.labelsize": 11,
        "axes.titlesize": 12, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 150,
    })
    fig, ax = plt.subplots(figsize=(11, 6))

    ds_rows = [r for r in rows if r["dataset"] == ds]
    for family in ("b200_pretrain", "b200_highres112", "b200_highres128"):
        pts = sorted(
            [(r["iteration"] + OFFSET[family], r["test_dice"])
             for r in ds_rows if r["family"] == family and r["test_dice"] is not None]
        )
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=FAMILY_COLOR[family], label=FAMILY_LABEL[family],
                 linewidth=1.5, markersize=5, alpha=0.9)

    h100 = next((r["test_dice"] for r in ds_rows if r["family"] == "h100_best"), None)
    if h100 is not None:
        ax.axhline(h100, color="#E64B35", linestyle="--", linewidth=1.5,
                    label=f"H100 best (9374): {h100:.3f}")

    rand = next((r["test_dice"] for r in ds_rows if r["family"] == "random_init"), None)
    if rand is not None:
        ax.axhline(rand, color="gray", linestyle=":", linewidth=1.5,
                    label=f"Random init: {rand:.3f}")

    ymax = ax.get_ylim()[1]
    ax.axvline(PRETRAIN_TOTAL_ITERS, color="dimgray", linestyle=":", linewidth=1.2)
    ax.text(PRETRAIN_TOTAL_ITERS, ymax * 0.99, " High-res adaptation starts (112³ & 128³ branch here)",
             ha="left", va="top", fontsize=8, color="dimgray")

    ax.set_xlabel("Cumulative training iteration (pretrain, then high-res 112³/128³ branching off it)",
                   fontweight="medium")
    ax.set_ylabel("Test Dice (linear probe)", fontweight="medium")
    ax.set_title(f"Linear Probing — {DATASET_SHORT.get(ds, ds)}", fontweight="bold", pad=10)
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="gray", framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved plot: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_results(args.base_dir)
    print(f"Loaded {len(rows)} completed runs")

    write_csv(rows, os.path.join(args.out_dir, "lp_comparison_summary.csv"))

    for ds in DATASETS:
        out_path = os.path.join(args.out_dir, f"lp_comparison_{DATASET_SHORT.get(ds, ds)}.jpg")
        plot_dataset(rows, ds, out_path)


if __name__ == "__main__":
    main()
