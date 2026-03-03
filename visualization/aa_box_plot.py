"""
This code is written by Ahmadreza Attarpour (a.attarpour@mail.utoronto.ca) 
It recieves the inputs from user without a any flag
the inputs should be .pickle files containing dice, recall, prec, f1, and hd for multiple image patches

usage example:

python aa_box_plot.py -i algo1_vs_gt_test.pickle algo2_vs_gt_test.pickle algo1_vs_gt_unseen.pickle algo2_vs_gt_unseen.pickle \
                      -o ./results
                      -a algo1 algo2 algo1 algo2
                      -d test test unseen unseen
                      -n name of the fig
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse
from scipy import stats

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
def load_pickle_data(file_path):
    """Loads pickle data and checks its validity."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # old: expected 5 items (dice, recall, prec, f1, hd)
        if len(data) != 2:
            raise ValueError("The pickle file must contain two items: dice, hd95.")
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        raise

def create_output_dir(out_dir):
    """Creates output directory if it doesn't exist."""
    os.makedirs(out_dir, exist_ok=True)

def perform_statistical_test(df, metrics, test_type):
    """Performs statistical tests (t-test or Mann-Whitney) based on the user input."""
    results = {}
    algorithms = df['Algorithm'].unique()
    data_types = df['data'].unique()
    
    for metric in metrics: 
        for i, alg1 in enumerate(algorithms):
            
            for j,  alg2 in enumerate(algorithms[i+1:], i+1):

                for data in data_types:
                    
                    data1 = df[df['Algorithm'] == alg1][df['data'] == data][metric]
                    data2 = df[df['Algorithm'] == alg2][df['data'] == data][metric]
                    #print(data1, data2)
                
                    if test_type == 'ttest':
                        stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                    elif test_type == 'mannwhitney':
                        stat, p_value = stats.mannwhitneyu(data1, data2)
                    else:
                        raise ValueError(f"Unsupported test type: {test_type}")
                    
                    results[(alg1, alg2, metric, data)] = p_value
    
    return results

def set_paper_style():
    """Set MICCAI publication-style aesthetics (Times New Roman, matching other CryoET figures)."""
    plt.rcParams.update({
        # old: 'font.family': 'sans-serif'
        # old: 'font.sans-serif': ['Arial', 'Helvetica']
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 7,                 # Base font size
        'axes.labelsize': 7,            # Axis label size
        'axes.titlesize': 8,            # Axis title size
        'axes.titleweight': 'bold',     # old: 'normal'
        'xtick.labelsize': 7,           # old: 6
        'ytick.labelsize': 7,           # old: 6
        'legend.fontsize': 7,           # old: 6
        'figure.figsize': (8, 6),
        'lines.linewidth': 1,
        'lines.markersize': 4,
        'axes.spines.top': False,       # cleaner look for publication
        'axes.spines.right': False,
    })



def generate_boxplots(df, metrics, out_dir, palette, add_dots, test_type, p_values=None, name=None):
    """Generates boxplots for specified metrics."""
    set_paper_style() 
    fig, axs = plt.subplots(ncols=len(metrics), figsize=(17, 5.5))
    
    for i, metric in enumerate(metrics):

        # Create boxplot
        sns.boxplot(x='data', y=metric, data=df, hue='Algorithm', ax=axs[i], palette=palette, showfliers=False)
        
        # Optionally add dots (strip plot)
        if add_dots:
            # Lighten colors and ensure they stay in 0-1 range
            adjusted_palette = []
            for c in palette:
                # Lighten the color
                lighter = sns.set_hls_values(c, l=0.3)  # Increase lightness
                # Clamp to valid range
                clamped = tuple(min(max(x, 0), 1) for x in lighter)
                adjusted_palette.append(clamped)

            sns.swarmplot(x='data', y=metric, data=df, hue='Algorithm',
                          ax=axs[i], palette=adjusted_palette,
                          dodge=True, alpha=0.7, linewidth=0.5)
            axs[i].get_legend().set_visible(False)

        if metric != "HD95":
            axs[i].set_ylim(bottom=0, top=1.05)
            axs[i].set_yticks(np.arange(0, 1.01, 0.1))
            # axs[i].spines['top'].set_visible(False)

        else:
            axs[i].set_yticks(np.arange(0, 220, 20))


        # Customize plot appearance
        axs[i].set_title(metric, fontsize=16)
        axs[i].xaxis.set_tick_params(labelsize=14)
        axs[i].yaxis.set_tick_params(labelsize=14)
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')
        #axs[i].yaxis.grid(True) # grid line

        # add legend to only one of the views
        #if i != 2:
        axs[i].legend([],[], frameon=False)
        # else:
        #     axs[i].legend(loc = 'lower left')

    plt.tight_layout()
    if name is not None:
        outname = f"{name}"
    else:
        outname = "box_plot"
    plt.savefig(os.path.join(out_dir, outname), dpi=300)
    plt.savefig(os.path.join(out_dir, outname + ".pdf"), format='pdf')
    plt.savefig(os.path.join(out_dir, outname + ".eps"), format='eps')
    plt.savefig(os.path.join(out_dir, outname + ".svg"), format='svg')

    print(f"Box plots saved to {out_dir}/{outname}")

# -------------------------------------------------------
# Main Function
# -------------------------------------------------------
def main(args):
    # Parse arguments
    input_paths = args['inputs']
    algorithm_names = args['algorithm_name']
    data_names = args['data_type']
    out_dir = args['out_dir']
    add_dots = args['add_dots']
    test_type = args.get('test_type', None)  # Default is None if not passed
    name = args['name']
    
    # Validate inputs
    if not (len(input_paths) == len(algorithm_names) == len(data_names)):
        raise ValueError("The number of inputs, algorithm names, and data types must match.")
    
    # Create output directory
    create_output_dir(out_dir)

    # Initialize data lists
    # old: dice_list, hd95_list, recall_list, prec_list, f1_list = [], [], [], [], []
    dice_list, hd95_list = [], []

    # Load data
    for path in input_paths:
        try:
            dice, hd95 = load_pickle_data(path)
            print("------------------------")
            print(f"{path}: mean +/- std dice:  {np.mean(dice):.4f} +/- {np.std(dice):.4f}")
            print(f"{path}: mean +/- std hd95:  {np.mean(hd95):.4f} +/- {np.std(hd95):.4f}")
            dice_list.append(dice)
            hd95_list.append(hd95)
        except Exception as e:
            print(f"Skipping file {path} due to an error: {e}")

    # Create DataFrame
    repeats = [len(dice) for dice in dice_list]
    data = {
        'Algorithm': [algorithm_names[i] for i, count in enumerate(repeats) for _ in range(count)],
        'data': [data_names[i] for i, count in enumerate(repeats) for _ in range(count)],
        'Dice Coefficient': [item for sublist in dice_list for item in sublist],
        'HD95': [item for sublist in hd95_list for item in sublist],
        # old: 'Precision', 'Recall', 'F1' removed
    }
    df = pd.DataFrame(data)
    print(df)

    # Set color palette
    unique_algorithms = list(set(algorithm_names))
    palette = sns.color_palette("pastel", n_colors=len(unique_algorithms)+3)
    # index 2 --> mapl3
    # index 1 ---> mapl3 scratch
    # index 0 ---> unetr
    # index 3 ---> TrailMap
    # index 4 ---> Dlmbmap
    palette = [palette[2], palette[3], palette[4]]
    print(palette)

    # stats
    # old: metric = ['Dice Coefficient', 'HD95', 'Precision', 'Recall']
    metric = ['Dice Coefficient', 'HD95']
    if test_type:
        p_values = perform_statistical_test(df, metric, test_type)
        print(f"p values: {p_values}")
    else:
        p_values = None


    # Generate plots
    generate_boxplots(df, metric, out_dir, palette, add_dots, test_type, p_values, name)

# -------------------------------------------------------
# Entry Point
# -------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flexible Box Plot Generator')
    parser.add_argument('-o', '--out_dir', help='Path of output directory', required=True)
    parser.add_argument('-i', '--inputs', nargs='+', help='Input file paths (one or more)', required=True)
    parser.add_argument('-a', '--algorithm_name', nargs='+', help='Algorithm names (one or more)', required=True)
    parser.add_argument('-d', '--data_type', nargs='+', help='Data names (one or more)', required=True)
    parser.add_argument('-n', '--name', help='name of the png to be saved', required=False, default=None)
    parser.add_argument('--add_dots', action='store_true', help='Add dots to the box plots (strip plots)')
    parser.add_argument('--test_type', choices=['ttest', 'mannwhitney'], help='Statistical test to perform between algorithms (t-test or Mann-Whitney)')
    args = vars(parser.parse_args())
    main(args)
