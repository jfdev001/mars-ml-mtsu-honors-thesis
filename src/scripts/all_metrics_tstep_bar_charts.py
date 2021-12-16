"""Script for computing the multi-ax bar chart per timestep for a model.

Takes a raw file consisting of columns of the format:

<Metric>_<timestep specification>

```python
python all_metrics_tstep_bar_charts.py \
    ../../logs/testing/unscaled/20211116_19-27-20CST_raw_fold_results-base_conv.csv \
    ../../logs/junk/multi_tstep_cnn.svg \
    --timestep_only False \
    --bar_width 0.2 \
    --sort ascending \
    --label_bar_mean_value True \
    --mean_value_text_align right \
    --x_mod_preset left
```

where the brackets are not included but represent arguments.

On the use of multiple bars in plt.bar
https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
"""

from __future__ import annotations

import argparse
from distutils.util import strtobool
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.container import BarContainer

from matplotlib.axes import Axes
import numpy as np
import pandas as pd

if not os.path.join(os.getcwd(), '..') in sys.path:
    sys.path.append(os.path.join(os.getcwd(), '..'))

import utils  # nopep8

if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser('script for plotting multi-bar metrics.')

    parser.add_argument(
        'read_path',
        help='path to read csv with one sample per cv-fold \
        (e.g., raw data).',
        type=str)

    parser.add_argument('write_path', help='path to write svg', type=str)

    parser.add_argument(
        '--metrics',
        choices=[
            'MeanAbsoluteError',
            'RootMeanSquaredError',
            'MeanSquaredError'],
        nargs='+',
        help='metrics to plot. (default: all available metrics in file)',
        default=None)

    parser.add_argument('--x_label', help='x-axis label', default='')

    parser.add_argument(
        '--bar_width',
        help='value to add to subsequent bars in multi-bar chart. (default: 0.1)',
        type=float,
        default=0.1)

    parser.add_argument('--y_label', help='y-axis label.', default='')

    parser.add_argument('--title', help='title for plot', default=None)

    parser.add_argument(
        '--split_col',
        help='str to split df column on for xtick labels \
            (e.g., `col.split(split_col)[-1]`)',
        type=str,
        default='_')

    parser.add_argument(
        '--timestep_only',
        choices=[True, False],
        help='true to include timestep xtick labels only, false otherwise. \
            Only applicable when reading from a single model`s raw data file. \
            (default: None)',
        type=lambda x: bool(strtobool(x)),
        default=True)

    parser.add_argument(
        '--sort',
        choices=['ascending', 'descending'],
        help='whether to sort elements of chart or not based on some metric. (default: None)',
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
        '--alpha', help='confidence level. (default: 0.95)', type=float, default=0.95)

    args = parser.parse_args()

    # Load mpl styling
    plt.style.use('report.mplstyle')

    # Load data
    df = pd.read_csv(args.read_path)

    # Columns in dataframe
    columns = df.columns

    # Determine whether to include timestep and overall
    # Exclude any column that cannot be mapped to int if desired
    if args.timestep_only:
        print('Time step only')
        columns = [col for col in columns if col.split(
            args.split_col)[-1].isdigit()]

    # Get metrics labels a set based on the column delimeter...
    # sets default value
    if args.metrics is None:

        # Without sorted, the set will be different each time
        metrics = sorted(list(set([col.split(args.split_col)[0]
                                   for col in columns])))
    else:
        metrics = args.metrics

    # Get xtick labels... there will be repeats e.g., there are multiple
    # timesteps per metric
    repeat_xtick_labels = [col.split(args.split_col)[-1]
                           for col in columns]

    # Get a sorted list of labels
    xtick_labels = np.array(sorted(list(set(repeat_xtick_labels))))

    # Compute the indices (x-position) of the bars for the mpl artist
    xpos = np.arange(len(xtick_labels))

    # Extract slices of dataframe based on metrics
    metric_dfs = [
        df[[col for col in columns
           if col.split(args.split_col)[0] == metric]] for metric in metrics]

    # Compute the appropriate data for the bar charts
    bar_means = []
    bar_errs = []
    for metric_df in metric_dfs:

        # E.g., MeanAbsoluteError_1, MeanAbsoluteError_2 ...
        metric_df_columns = metric_df.columns

        metric_means = [np.mean(metric_df[col]) for col in metric_df_columns]
        metric_errs = [utils.confidence_interval_err(
            metric_df[col], alpha=args.alpha) for col in metric_df_columns]

        bar_means.append(metric_means)
        bar_errs.append(metric_errs)

    # Cast data lists to numpy
    bar_means = np.array(bar_means)
    bar_errs = np.array(bar_errs)

    # Check sorting
    if args.sort is not None:

        # Log
        print(f'Sorting in {args.sort} order...')

        # Extract the first metrics means
        # NOTE: Could use key:value pairs for metric_name: means
        # or metric_name: errs
        means = bar_means[0]

        # Determine order
        if args.sort == 'descending':
            sort_ixs = np.argsort(-1*means)

        elif args.sort == 'ascending':
            sort_ixs = np.argsort(means)

        # Re-sort all the metrics in the bar lists
        for ix in range(len(bar_means)):
            bar_means[ix] = bar_means[ix][sort_ixs]
            bar_errs[ix] = bar_errs[ix][sort_ixs]

        # Sort the timestep indices
        xtick_labels = xtick_labels[sort_ixs]

    # Plotting
    fig, ax = plt.subplots()

    # Plot each of the rectangle objects for each metric
    barcontainers = []
    for ix, (means, errs, metric) in enumerate(zip(bar_means, bar_errs, metrics)):
        barcontainer = ax.bar(
            xpos+args.bar_width*ix,
            means,
            yerr=errs,
            ecolor='black',
            width=args.bar_width,
            alpha=0.5,
            capsize=10,
            label=metric)

        barcontainers.append(barcontainer)

    # Set generic labels first for
    ax.set_xlabel(args.x_label)
    ax.set_xticks(xpos+args.bar_width)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel(args.y_label)
    ax.set_title(args.title)
    ax.legend()
    ax.yaxis.grid(True)

    # Label the bar value
    if args.label_bar_mean_value:

        # Set modifier
        if args.x_mod_custom is not None:
            x_mod = args.x_mod_custom
        else:
            x_mod = args.x_mod_preset

        for barcontainer in barcontainers:
            utils.autolabel(
                ax,
                x_mod=x_mod,
                ha=args.mean_value_text_align,
                barcontainer=barcontainer)

    # Save fig
    print('Saving fig to', args.write_path)
    fig.savefig(args.write_path, bbox_inches='tight')
