"""
Line plot of average foreground Dice vs. training data percentage.
Compares CryoDINO (Frozen + ViTAdapter) vs. nnU-Net for DS-10001 and DS-10010.

CryoDINO values sourced from:
    dataset/downstream/finetuning/results.md — Frozen + ViTAdapter rows

nnU-Net values sourced from per-case inference results:
    DS-10001: mean(class_1, class_2, class_3) averaged over TS_0003 and TS_0009
    DS-10010: class_1 averaged over TE13, TE14, UE4, UF4, UF6

Usage:
    python aa_fg_dice_vs_percent.py -o ./results -n fg_dice_vs_percent
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


# ── CryoDINO (Frozen + ViTAdapter) — from results.md ─────────────────────────
# DS-10001: fg = mean(cls1, cls2, cls3) from per-class dice [bg, cls1, cls2, cls3]
# DS-10010: fg = cls1
CRYO = {
    "DS-10001": {
        10:  np.mean([0.92, 0.78, 0.72]),   # [0.99, 0.92, 0.78, 0.72]
        50:  np.mean([0.95, 0.86, 0.73]),   # [0.99, 0.95, 0.86, 0.73]
        100: np.mean([0.96, 0.89, 0.72]),   # [0.99, 0.96, 0.89, 0.72]
    },
    "DS-10010": {
        10:  0.64,   # [1.00, 0.64]
        50:  0.67,   # [1.00, 0.67]
        100: 0.64,   # [1.00, 0.64]
    },
}


# ── nnU-Net — per-case inference results ──────────────────────────────────────
def _fg_avg_001(cases):
    """Average foreground dice for DS001: mean(cls1,cls2,cls3) per case, then mean over cases."""
    return float(np.mean([np.mean([c1, c2, c3]) for c1, c2, c3 in cases]))


NN = {
    "DS-10001": {
        10:  _fg_avg_001([(0.932787716, 0.840065658, 0.830084026),   # TS_0003
                           (0.898028314, 0.640168250, 0.588857532)]),  # TS_0009
        50:  _fg_avg_001([(0.969907403, 0.934249640, 0.842424452),
                           (0.946348071, 0.861892164, 0.588312209)]),
        100: _fg_avg_001([(0.972111762, 0.935617268, 0.831301868),
                           (0.950243771, 0.865654647, 0.583852828)]),
    },
    "DS-10010": {
        10:  float(np.mean([0.684726477, 0.569212139, 0.718274176, 0.621484518, 0.598034799])),
        50:  float(np.mean([0.691031992, 0.591504276, 0.764116347, 0.636706829, 0.653793395])),
        100: float(np.mean([0.672070086, 0.509247243, 0.702344239, 0.478048325, 0.702472389])),
    },
}

PERCENTS = [10, 50, 100]


def set_paper_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8.5,
        'lines.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def main(out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    set_paper_style()

    palette = sns.color_palette("tab10", n_colors=2)
    colors  = {"DS-10001": palette[0], "DS-10010": palette[1]}
    markers = {"DS-10001": 'o', "DS-10010": 's'}

    fig, ax = plt.subplots(figsize=(5.5, 4))

    for ds in ["DS-10001", "DS-10010"]:
        c = colors[ds]
        m = markers[ds]
        ys_cryo = [CRYO[ds][p] for p in PERCENTS]
        ys_nn   = [NN[ds][p]   for p in PERCENTS]

        ax.plot(PERCENTS, ys_cryo, color=c, marker=m, markersize=6,
                linestyle='-',  label=f'CryoDINO — {ds}', zorder=3)
        ax.plot(PERCENTS, ys_nn,   color=c, marker=m, markersize=6,
                linestyle='--', label=f'nnU-Net — {ds}',  zorder=3, alpha=0.75)

    ax.set_xlabel("Training Data (%)", fontsize=10)
    ax.set_ylabel("Avg. Foreground Dice", fontsize=10)
    ax.set_title("CryoDINO vs. nnU-Net — Foreground Dice vs. Training Data",
                 fontsize=10, fontweight='bold')
    ax.set_xticks(PERCENTS)
    ax.set_xticklabels(["10%", "50%", "100%"])
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0.55, 0.95)
    ax.legend(frameon=False, ncol=1)

    plt.tight_layout()

    out_base = os.path.join(out_dir, name)
    plt.savefig(out_base + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(out_base + ".pdf", format='pdf', bbox_inches='tight')
    plt.savefig(out_base + ".svg", format='svg', bbox_inches='tight')
    print(f"Saved to {out_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', default='visualization/results')
    parser.add_argument('-n', '--name',    default='fg_dice_vs_percent')
    args = parser.parse_args()
    main(args.out_dir, args.name)
