"""Script to write the lower, upper, and mean of raw data.

Essentially the same as the bar charts script except without the bar
charts and writes csv instead.

```python
python confidence_intervals_bounds.py \
    ../../logs/report/mae_models_fold_overall.csv \
    ../../logs/report/mae_ci.csv \
    --split_col - \
    --sort ascending \
    --transpose True
```
"""


import argparse
from distutils.util import strtobool
import os
import sys

import numpy as np
import pandas as pd


if not os.path.join(os.getcwd(), '..') in sys.path:
    sys.path.append(os.path.join(os.getcwd(), '..'))

from utils import confidence_interval_err  # nopep8

if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser(
        'build table for single metric`s +- conf-interval err.')

    parser.add_argument(
        'read_path',
        help='path to read csv that has columns of form <metric>-<model_name> \
        with one sample per cv-fold (e.g., raw data).',
        type=str)

    parser.add_argument('write_path', help='path to write svg', type=str)

    parser.add_argument(
        '--metric',
        help='metric (`MeanAbsoluteError`, \
            `RootMeanSquaredError`,\
            `MeanSquaredError`) or other substr to extract from file. (default: MeanAbsoluteError)',
        type=str,
        default='MeanAbsoluteError',)

    parser.add_argument(
        '--split_col',
        help='str to split df column on for xtick labels \
            (e.g., `col.split(split_col)[-1]`)',
        type=str,
        default=None)

    parser.add_argument(
        '--alpha', help='confidence level. (default: 0.95)', type=float, default=0.95)

    parser.add_argument(
        '--sort',
        choices=['ascending', 'descending'],
        help='whether to sort elements of chart or not. (default: None)',
        default=None)

    parser.add_argument(
        '--transpose',
        choices=[True, False],
        help='true to transpose data frame (i.e., header: models -> bounds). \
        (default: False)',
        type=lambda x: bool(strtobool(x)),
        default=False)

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.read_path)

    # Load only those columns that have the desired metric
    columns = [column for column in df.columns if column.find(
        args.metric) != -1]

    df = df[columns]

    # Set x tick labels
    if args.split_col is not None:
        print('Splitting df columns on:', args.split_col)
        xtick_labels = np.array(
            [col.split(args.split_col)[-1] for col in columns])
    else:
        print('Not splitting columnns...')
        xtick_labels = np.array(columns)

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

    # The dataframe to store the table of the form
    #       model1   model2  modelN
    # lower
    # mean
    # upper

    #  Compute the lower and upper bound for a single model and make
    # dictionary to be added to the write dataframe
    print('Exracting CI info...')
    df_dict = {}
    for ix in range(len(xtick_labels)):
        model_name = xtick_labels[ix]

        model_lower = means[ix] - conf_intervals[ix]
        model_mean = means[ix]
        model_upper = means[ix] + conf_intervals[ix]

        df_dict[model_name] = [model_lower, model_mean, model_upper]

    # Make a dataframe and label the indices
    conf_interval_df = pd.DataFrame(df_dict)
    conf_interval_df.index = ['lower', 'mean', 'upper']

    # Transpose or not
    if args.transpose:
        print('Transposing..')
        conf_interval_df = conf_interval_df.transpose()

    # Write the dataframe
    print('Writing to ', args.write_path)
    conf_interval_df.to_csv(args.write_path, index=True)
