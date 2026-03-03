"""
Plot training losses from training_metrics.json
Usage: python plot_training_losses.py --input training_metrics.json --output training_losses.jpg
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

# Nature Methods style color palette
NATURE_COLORS = {
    'total_loss': '#E64B35',        # Red
    'dino_local_crops_loss': '#4DBBD5',  # Cyan
    'dino_global_crops_loss': '#00A087', # Teal
    'koleo_loss': '#3C5488',        # Blue
    'ibot_loss': '#F39B7F',         # Salmon
}

LOSS_LABELS = {
    'total_loss': 'Total Loss',
    'dino_local_crops_loss': 'DINO Local Crops',
    'dino_global_crops_loss': 'DINO Global Crops',
    'koleo_loss': 'KoLeo Loss',
    'ibot_loss': 'iBOT Loss',
}


def load_metrics(filepath):
    """Load metrics from JSON lines file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def plot_losses(data, output_path, dpi=300):
    """Plot all losses over iterations."""
    # Set up Nature-style plotting
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.linewidth': 1.0,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
    })

    # Extract data
    iterations = [d['iteration'] for d in data]

    loss_keys = ['total_loss', 'dino_local_crops_loss', 'dino_global_crops_loss', 'koleo_loss', 'ibot_loss']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    for key in loss_keys:
        values = [d[key] for d in data]
        ax.plot(iterations, values,
                color=NATURE_COLORS[key],
                label=LOSS_LABELS[key],
                linewidth=1.5,
                alpha=0.9)

    ax.set_xlabel('Iteration', fontweight='medium')
    ax.set_ylabel('Loss', fontweight='medium')
    ax.set_title('Training Losses', fontweight='bold', pad=10)

    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='gray', framealpha=0.95)

    # Grid
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to training_metrics.json')
    parser.add_argument('--output', type=str, default='training_losses.jpg', help='Output image path')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    args = parser.parse_args()

    data = load_metrics(args.input)
    print(f"Loaded {len(data)} data points")

    plot_losses(data, args.output, args.dpi)


if __name__ == '__main__':
    main()
