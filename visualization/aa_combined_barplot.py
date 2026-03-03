"""
Combined bar plot for DS-10001, DS-10010, EMPIAR-10989 at 100% training data.
Compares CryoDINO (frozen pre-trained) vs nnU-Net (fully supervised).

Usage:
    python aa_combined_barplot.py -o ./results -n combined_barplot
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import argparse
import os


# ── Data ──────────────────────────────────────────────────────────────────────
DATASETS = {
    "DS-10001": {
        "classes": ["Cytoplasm", "Organelles", "Membrane"],
        "cryo_dice": {
            "Cytoplasm":  np.array([0.9674321413040161, 0.9425867199897766]),
            "Organelles": np.array([0.9306117296218872, 0.8444172143936157]),
            "Membrane":   np.array([0.8465653657913208, 0.5910652279853821]),
        },
        "cryo_hd95": {
            "Cytoplasm":  np.array([10.0,              124.05240631103516]),
            "Organelles": np.array([8.782063484191895,  23.76972770690918]),
            "Membrane":   np.array([7.348469257354736,  85.09406280517578]),
        },
        "nn_dice": {
            "Cytoplasm":  np.array([0.972111762,  0.950243771]),
            "Organelles": np.array([0.935617268,  0.865654647]),
            "Membrane":   np.array([0.831301868,  0.583852828]),
        },
        "nn_hd95": {
            "Cytoplasm":  np.array([17.52141571, 97.88258362]),
            "Organelles": np.array([6.708203793, 49.25063324]),
            "Membrane":   np.array([6.0,         88.01704407]),
        },
    },
    "DS-10010": {
        "classes": ["Membrane"],
        "cryo_dice": {"Membrane": np.array([0.5563250780105591, 0.5143744945526123,
                                             0.7635380029678345, 0.7065274715423584,
                                             0.6823450326919556])},
        "cryo_hd95": {"Membrane": np.array([169.35169982910156, 548.5453491210938,
                                             354.2724914550781,  10.049875259399414,
                                             101.65567016601562])},
        # "cryo_dice": {"Membrane": np.array([0.5063273906707764, 0.5461863279342651,
        #                                      0.7420262694358826, 0.6961358785629272,
        #                                      0.6885470151901245])},
        # "cryo_hd95": {"Membrane": np.array([164.5782470703125, 548.8187255859375,
        #                                      357.7121887207031,  8.124038696289062,
        #                                      108.00926208496094])},
        "nn_dice":   {"Membrane": np.array([0.509247243, 0.672070086, 0.702344239,
                                             0.478048325, 0.702472389])},
        "nn_hd95":   {"Membrane": np.array([179.0977325, 111.950882,  360.1749573,
                                             555.4918213, 141.3187866])},
    },
    "EMPIAR-10989": {
        "classes": ["Actin Filaments"],
        "cryo_dice": {"Actin Filaments": np.array([0.3071155548095703])},
        "cryo_hd95": {"Actin Filaments": np.array([26.795522689819336])},
        "nn_dice":   {"Actin Filaments": np.array([0.024651486])},
        "nn_hd95":   {"Actin Filaments": np.array([133.8394623])},
    },
}


def set_paper_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 1,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_dataset_row(axs_row, ds_name, ds_data, color_cryo, color_nn):
    classes    = ds_data["classes"]
    cryo_dice  = ds_data["cryo_dice"]
    cryo_hd95  = ds_data["cryo_hd95"]
    nn_dice    = ds_data["nn_dice"]
    nn_hd95    = ds_data["nn_hd95"]

    x = np.arange(len(classes))
    w = 0.35

    for ax, d_cryo, d_nn, metric in zip(
            axs_row,
            [cryo_dice, cryo_hd95],
            [nn_dice,   nn_hd95],
            ["Dice Coefficient", "HD95"]):

        means_c = [np.mean(d_cryo[c]) for c in classes]
        stds_c  = [np.std(d_cryo[c])  for c in classes]
        means_n = [np.mean(d_nn[c])   for c in classes]
        stds_n  = [np.std(d_nn[c])    for c in classes]

        # Only show error bars if more than 1 sample
        eb_c = stds_c if len(list(d_cryo.values())[0]) > 1 else None
        eb_n = stds_n if len(list(d_nn.values())[0])   > 1 else None

        ax.bar(x - w/2, means_c, w, yerr=eb_c, color=color_cryo,
               capsize=3, error_kw={'linewidth': 0.8}, zorder=3)
        ax.bar(x + w/2, means_n, w, yerr=eb_n, color=color_nn,
               capsize=3, error_kw={'linewidth': 0.8}, zorder=3)

        # Individual data points
        dot_c = tuple(min(max(v, 0), 1) for v in sns.set_hls_values(color_cryo, l=0.35))
        dot_n = tuple(min(max(v, 0), 1) for v in sns.set_hls_values(color_nn,   l=0.35))
        for i, cls in enumerate(classes):
            ax.scatter(np.full(len(d_cryo[cls]), i - w/2), d_cryo[cls],
                       color=dot_c, zorder=4, s=14, alpha=0.85, edgecolors='none')
            ax.scatter(np.full(len(d_nn[cls]),   i + w/2), d_nn[cls],
                       color=dot_n, zorder=4, s=14, alpha=0.85, edgecolors='none')

        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        all_vals = [v for c in classes for v in list(d_cryo[c]) + list(d_nn[c])]
        rng = max(all_vals) - min(all_vals) if max(all_vals) != min(all_vals) else max(all_vals)
        ax.set_ylim(max(0, min(all_vals) - 0.12 * rng),
                    max(all_vals) + 0.18 * rng)

    # Dataset label on the left of the row
    axs_row[0].set_ylabel(ds_name, fontsize=10, fontweight='bold', labelpad=8)


def main(out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    set_paper_style()

    palette    = sns.color_palette("pastel", n_colors=5)
    color_cryo = palette[2]
    color_nn   = palette[3]

    n_rows = len(DATASETS)
    fig, axs = plt.subplots(n_rows, 2, figsize=(9, 3.5 * n_rows))

    for row_idx, (ds_name, ds_data) in enumerate(DATASETS.items()):
        plot_dataset_row(axs[row_idx], ds_name, ds_data, color_cryo, color_nn)

    # Column titles on top row only
    axs[0][0].set_title("Dice Coefficient", fontsize=12, fontweight='bold')
    axs[0][1].set_title("HD95",             fontsize=12, fontweight='bold')

    # Single legend at top
    handles = [
        mpatches.Patch(color=color_cryo, label='CryoDINO (Frozen pre-trained feature model)'),
        mpatches.Patch(color=color_nn,   label='nnU-Net (Fully supervised)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_base = os.path.join(out_dir, name)
    plt.savefig(out_base + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(out_base + ".pdf", format='pdf', bbox_inches='tight')
    plt.savefig(out_base + ".svg", format='svg', bbox_inches='tight')
    print(f"Saved to {out_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', default='visualization/results')
    parser.add_argument('-n', '--name',    default='combined_barplot')
    args = parser.parse_args()
    main(args.out_dir, args.name)
