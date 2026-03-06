"""
Combined bar plot for DS-10001, DS-10010, EMPIAR-10989 at 100% training data.
Compares CryoDINO (frozen pre-trained) vs nnU-Net (fully supervised) vs Random Init.

Usage:
    python aa_combined_barplot.py -o ./results -n combined_barplot_with_random_init
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
        # Random init (ViTAdapterUNETR, fully supervised, no pretraining) — log j1_5872616
        "rand_dice": {
            "Cytoplasm":  np.array([0.9526899456977844, 0.9213135838508606]),
            "Organelles": np.array([0.9053843021392822, 0.7346742749214172]),
            "Membrane":   np.array([0.8261680603027344, 0.5938450098037720]),
        },
        "rand_hd95": {
            "Cytoplasm":  np.array([33.6749153137207,   114.63856506347656]),
            "Organelles": np.array([19.595918655395508, 147.2480926513672]),
            "Membrane":   np.array([9.165151596069336,   82.75868225097656]),
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
        # Random init — log j1_5872616 (TE14, UF4, UE4, UF6, TE13)
        "rand_dice": {"Membrane": np.array([0.4738955795764923, 0.4473776817321777,
                                             0.7381826639175415, 0.6912893056869507,
                                             0.6736786365509033])},
        "rand_hd95": {"Membrane": np.array([184.19012451171875, 552.9828491210938,
                                             358.33642578125,    152.728515625,
                                             127.88275909423828])},
    },
    "EMPIAR-10989": {
        "classes": ["Actin Filaments"],
        "cryo_dice": {"Actin Filaments": np.array([0.3171155548095703])},
        "cryo_hd95": {"Actin Filaments": np.array([26.795522689819336])},
        "nn_dice":   {"Actin Filaments": np.array([0.024651486])},
        "nn_hd95":   {"Actin Filaments": np.array([133.8394623])},
        # Random init — log j2_5872701 (00011)
        "rand_dice": {"Actin Filaments": np.array([0.31524153661727905])},
        "rand_hd95": {"Actin Filaments": np.array([35.0])},
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


def plot_dataset_col(ax, ds_data, metric_key, classes, color_cryo, color_nn, color_rand):
    """Plot one dataset × one metric into a single axis.
    Bar order: Random Init | nnU-Net | CryoDINO
    """
    d_rand = ds_data[f"rand_{metric_key}"]
    d_nn   = ds_data[f"nn_{metric_key}"]
    d_cryo = ds_data[f"cryo_{metric_key}"]

    x = np.arange(len(classes))
    w = 0.25

    means_r = [np.mean(d_rand[c]) for c in classes]
    stds_r  = [np.std(d_rand[c])  for c in classes]
    means_n = [np.mean(d_nn[c])   for c in classes]
    stds_n  = [np.std(d_nn[c])    for c in classes]
    means_c = [np.mean(d_cryo[c]) for c in classes]
    stds_c  = [np.std(d_cryo[c])  for c in classes]

    eb_r = stds_r if len(list(d_rand.values())[0]) > 1 else None
    eb_n = stds_n if len(list(d_nn.values())[0])   > 1 else None
    eb_c = stds_c if len(list(d_cryo.values())[0]) > 1 else None

    ax.bar(x - w, means_r, w, yerr=eb_r, color=color_rand,
           capsize=3, error_kw={'linewidth': 0.8}, zorder=3)
    ax.bar(x,     means_n, w, yerr=eb_n, color=color_nn,
           capsize=3, error_kw={'linewidth': 0.8}, zorder=3)
    ax.bar(x + w, means_c, w, yerr=eb_c, color=color_cryo,
           capsize=3, error_kw={'linewidth': 0.8}, zorder=3)

    dot_r = tuple(min(max(v, 0), 1) for v in sns.set_hls_values(color_rand, l=0.35))
    dot_n = tuple(min(max(v, 0), 1) for v in sns.set_hls_values(color_nn,   l=0.35))
    dot_c = tuple(min(max(v, 0), 1) for v in sns.set_hls_values(color_cryo, l=0.35))
    for i, cls in enumerate(classes):
        ax.scatter(np.full(len(d_rand[cls]), i - w), d_rand[cls],
                   color=dot_r, zorder=4, s=14, alpha=0.85, edgecolors='none')
        ax.scatter(np.full(len(d_nn[cls]),   i),     d_nn[cls],
                   color=dot_n, zorder=4, s=14, alpha=0.85, edgecolors='none')
        ax.scatter(np.full(len(d_cryo[cls]), i + w), d_cryo[cls],
                   color=dot_c, zorder=4, s=14, alpha=0.85, edgecolors='none')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    all_vals = [v for c in classes
                for v in list(d_rand[c]) + list(d_nn[c]) + list(d_cryo[c])]
    rng = max(all_vals) - min(all_vals) if max(all_vals) != min(all_vals) else max(all_vals)
    ax.set_ylim(max(0, min(all_vals) - 0.12 * rng),
                max(all_vals) + 0.18 * rng)


def main(out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    set_paper_style()

    palette     = sns.color_palette("pastel", n_colors=6)
    color_cryo  = palette[2]
    color_nn    = palette[3]
    color_rand  = palette[4]

    ds_names = list(DATASETS.keys())
    n_cols   = len(ds_names)                        # 3 datasets → 3 columns
    # Layout: 2 rows (Dice top, HD95 bottom) × 3 columns (one per dataset)
    fig, axs = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 7))

    for col_idx, ds_name in enumerate(ds_names):
        ds_data = DATASETS[ds_name]
        classes = ds_data["classes"]

        plot_dataset_col(axs[0][col_idx], ds_data, "dice", classes, color_cryo, color_nn, color_rand)
        plot_dataset_col(axs[1][col_idx], ds_data, "hd95", classes, color_cryo, color_nn, color_rand)

        # Dataset name as column title
        axs[0][col_idx].set_title(ds_name, fontsize=11, fontweight='bold', pad=6)

    # Row labels on the leftmost column
    axs[0][0].set_ylabel("Dice Coefficient", fontsize=10, fontweight='bold', labelpad=8)
    axs[1][0].set_ylabel("HD95",             fontsize=10, fontweight='bold', labelpad=8)

    # Single legend at top
    handles = [
        mpatches.Patch(color=color_rand,  label='Random Init (Fully supervised, no pretraining)'),
        mpatches.Patch(color=color_nn,    label='nnU-Net (Fully supervised)'),
        mpatches.Patch(color=color_cryo,  label='CryoDINO (Frozen pre-trained feature model)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3,
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
    parser.add_argument('-n', '--name',    default='combined_barplot_with_random_init')
    args = parser.parse_args()
    main(args.out_dir, args.name)
