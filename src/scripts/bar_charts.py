"""Script for computing desired error bar chart for all models.

Should use RAW data (i.e., results from each fold and not summary
statistics from each fold).

```
# Computing bar charts for all models
python bar_charts.py     \
    ../../logs/report/rmse_models_fold_overall.csv     \
    ../../logs/report/rmse_models_overall_charts.svg  \
    --split_col "-" 
    --rotate_xtick_labels 45     \
    --sort ascending     \
    --metric RootMeanSquaredError    \
    --title "5-Fold Time Series Cross Validated Model RMSE" \
    --y_label "RMSE for Predicted Air Temp. (K)"  \
    --label_bar_mean_value True \
    --x_mod_custom -0.2 \
    --mean_value_text_align right
```

```
# Computing bar charts for base-conv model only considering timesteps and single
# metric
python bar_charts.py \
    ../../logs/testing/unscaled/20211116_19-27-20CST_raw_fold_results-base_conv.csv \
    ../../logs/junk/cnn_mae_tsteps.svg \
    --metric MeanAbsoluteError \
    --split_col _ \
    --timestep_only on
```

Considers only one metric ... though could be expanded to include
multiple metrics.
https://problemsolvingwithpython.com/06-Plotting-with-Matplotlib/06.07-Error-Bars/
https://www.statology.org/confidence-intervals-python/

Rotation:
https://stackabuse.com/rotate-axis-labels-in-matplotlib/
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import sys
import os
from distutils.util import strtobool
from pathlib import Path

if not os.path.join(os.getcwd(), '..') in sys.path:
    sys.path.append(os.path.join(os.getcwd(), '..'))

from utils import confidence_interval_err, autolabel  # nopep8

if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser('plotting bar charts with error')

    parser.add_argument(
        'read_path',
        help='path to read csv with one sample per cv-fold \
        (e.g., raw data).',
        type=str)

    parser.add_argument('write_path', help='path to write svg', type=str)

    parser.add_argument(
        '--metric',
        help='metric (`MeanAbsoluteError`, \
            `RootMeanSquaredError`,\
            `MeanSquaredError`) or other substr to extract from file plot. (default: MeanAbsoluteError)',
        type=str,
        default='MeanAbsoluteError')

    parser.add_argument('--x_label', help='x-axis label', default='')

    parser.add_argument('--y_label', help='y-axis label.', default='')

    parser.add_argument('--title', help='title for plot', default=None)

    parser.add_argument(
        '--split_col',
        help='str to split df column on for xtick labels \
            (e.g., `col.split(split_col)[-1]`)',
        type=str,
        default=None)

    parser.add_argument(
        '--timestep_only',
        choices=[True, False],
        help='true to include timestep xtick labels only, false otherwise. \
            Only applicable when reading from a single model`s raw data file. \
            (default: None)',
        type=lambda x: bool(strtobool(x)),
        default=None)

    parser.add_argument(
        '--rotate_xtick_labels',
        help='rotate xtick labels by a number. (default: None)',
        type=int,
        default=None)

    parser.add_argument(
        '--label_bar_mean_value',
        choices=[True, False],
        help='true to label the value of a bar, false otherwise. (default: False)',
        type=lambda x: bool(strtobool(x)),
        default=False)

    parser.add_argument(
        '--x_mod_preset',
        choices=['left', 'left_center', 'center', 'right_center', 'right'],
        help='modify position of text label if applicable',
        type=str,
        default='left')

    parser.add_argument(
        '--x_mod_custom',
        help='custom float to add/subtract to text position. (default: None)',
        type=float,
        default=None)

    parser.add_argument(
        '--mean_value_text_align',
        help='horizontal alignment for ax.text',
        type=str,
        default='left')

    parser.add_argument(
        '--sort',
        choices=['ascending', 'descending'],
        help='whether to sort elements of chart or not. (default: None)',
        default=None)

    parser.add_argument(
        '--alpha', help='confidence level. (default: 0.95)', type=float, default=0.95)

    args = parser.parse_args()

    # Load styling
    plt.style.use('report.mplstyle')

    # Default title
    if args.title is None:
        args.title = Path(args.read_path).name + ' ' + \
            str(args.alpha * 100) + ' % CI'

    # Load the csv
    print('Loading data...')
    df = pd.read_csv(args.read_path)
    print(df.describe())

    # Load only those columns that have the desired metric
    columns = [column for column in df.columns if column.find(
        args.metric) != -1]

    # Exclude any column that cannot be mapped to int if desired
    if args.timestep_only:
        columns = [col for col in columns if col.split(
            args.split_col)[-1].isdigit()]

    # Extract the desired columns from the dataframe
    df = df[columns]

    # Log columns
    print(columns)

    # Set x tick labels
    if args.split_col is not None:
        print('Splitting df columns on:', args.split_col)
        xtick_labels = np.array(
            [col.split(args.split_col)[-1] for col in columns])
    else:
        print('Not splitting columnns...')
        xtick_labels = np.array(columns)

    # Log labels
    print(*xtick_labels)

    # Compute means for a given column as well as the confidence interval
    # From the confidence interval derive the approximate standard deviation
    print('Means & conf intervals...')
    means = np.array([np.mean(df[col]) for col in columns])

    # Confidence intervals -- 95%
    print(f'Computing {args.alpha * 100}% confidence interval...')
    conf_intervals = np.array([confidence_interval_err(
        df[col], alpha=args.alpha) for col in columns])

    # Decide sorting or not
    if args.sort is not None:

        # Log
        print(f'Sorting in {args.sort} order...')

        # Determine order
        if args.sort == 'descending':
            sort_ixs = np.argsort(-1*means)
        elif args.sort == 'ascending':
            sort_ixs = np.argsort(means)

        # Sort all relevants arrays
        means = means[sort_ixs]
        conf_intervals = conf_intervals[sort_ixs]
        xtick_labels = xtick_labels[sort_ixs]

    # Plot these with the labels based on the column timestep
    x_pos = np.arange(len(xtick_labels))

    # Plotting
    print('Plotting...')
    fig, ax = plt.subplots()

    barcontainer = ax.bar(x_pos, means,
                          yerr=conf_intervals,
                          align='center',
                          alpha=0.5,
                          ecolor='black',
                          capsize=10)

    ax.set_xlabel(args.x_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xtick_labels)

    # Rotation if needed
    if args.rotate_xtick_labels is not None:
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=args.rotate_xtick_labels, ha='right')

    ax.set_ylabel(args.y_label)

    ax.set_title(args.title)

    ax.yaxis.grid(True)

    # Determine autolabeling
    if args.label_bar_mean_value:

     # Set modifier
        if args.x_mod_custom is not None:
            x_mod = args.x_mod_custom
        else:
            x_mod = args.x_mod_preset

        autolabel(ax=ax, barcontainer=barcontainer, x_mod=x_mod)

    print('Saving...')
    fig.savefig(args.write_path, bbox_inches='tight')
