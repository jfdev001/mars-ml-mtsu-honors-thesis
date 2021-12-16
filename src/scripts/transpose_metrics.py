"""Script for reformatting performance metrics that are single line-multimodel.

```python
python transpose_metrics.py \
    ../../logs/report/20211202_all_models_summary_mae_mean_overall.csv \
    ../../logs/report/20211202_all_models_summary_mse_mean_overall.csv \
    ../../logs/report/20211202_all_models_summary_rmse_mean_overall.csv \
    --write_path ../../logs/report/models_mae_mse_rmse.csv \
    --sort ascending \
    --header metrics
```
"""

from __future__ import annotations

import argparse

import pandas as pd
import numpy as np


def all_arrs_in_lst_equal(lst_of_arrs: list[np.ndarray[str]]) -> bool:
    """Pairwise comparison of all elements of list of arrays to see if equal."""

    for ix in range(len(lst_of_arrs)-1):
        cur_cols = lst_of_arrs[ix]
        next_cols = lst_of_arrs[ix+1]
        if not np.array_equal(cur_cols, next_cols):
            return False

    return True


if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser(
        'script for concatenating list into \
        dataframe and setting model names as columns')
    parser.add_argument(
        'read_paths',
        help='list of summary data files to combine into single csv',
        nargs='+')
    parser.add_argument(
        '--write_path',
        help='path to write concatenated df to.',
        required=True,
        type=str)
    parser.add_argument(
        '--model_substr',
        help='delimiter for the model name. (default: -)',
        type=str,
        default='-')
    parser.add_argument(
        '--metric_substr',
        help='delimiter for metric name. (default: _)',
        type=str,
        default='_')
    parser.add_argument(
        '--sort',
        choices=['ascending', 'descending'],
        help='whether to sort elements of chart or not. \
            Only good if you know ahead of time that all metrics are \
            proportional (e.g., if MSE is low RMSE is low). (default: None)',
        default=None)
    parser.add_argument(
        '--header',
        choices=['metrics', 'models'],
        help='determines the header for the csv. (default: models)',
        default='models')
    args = parser.parse_args()

    # Initialize the parent dataframe
    parent_df = pd.DataFrame()

    # Load all data
    dfs = [pd.read_csv(path) for path in args.read_paths]

    # Validate that dataframes have only a single row
    for df in dfs:
        if df.shape[0] > 1:
            raise ValueError('dataframes must have only a single row.')

    # Extract the model names based on the columns in the csvs
    all_model_names = [np.array([col.split(args.model_substr)[-1]
                                 for col in df.columns]) for df in dfs]

    # Check whther transposition can occur
    if all_arrs_in_lst_equal(lst_of_arrs=all_model_names):
        model_names = all_model_names[0]
    else:
        raise ValueError(
            'transposition cannot occur for metrics that are for different models.')

    # Sort data assuming correlation among metrics and single row
    if args.sort is not None:
        metric_vector = dfs[0].iloc[0]

        # Indicate what is sorted along
        print('Sorted along:', dfs[0].columns[0].split(args.metric_substr)[0])

        sorted_ixs = np.argsort(metric_vector)
        if args.sort == 'descending':
            sorted_ixs = np.flip(sorted_ixs)

        # Use the sorted indices to sorted both values and columns of dfs
        model_names = model_names[sorted_ixs]
        for df in dfs:
            df.iloc[0] = df.iloc[0][sorted_ixs]

    # Determine metric ordering
    # for metric in dfs:
    #  if metric not in args.metrics:
    #    raise
    metrics = [df.columns[0].split(args.metric_substr)[0] for df in dfs]

    # Final procesing
    for df in dfs:

        # Change all the columns of the dfs to the model_names
        df.columns = model_names

        # Append all dfs to the parent df
        parent_df = parent_df.append(df)

    # Set the indices
    parent_df.index = metrics

    # Determine transposition
    if args.header == 'metrics':
        parent_df = parent_df.transpose()

    # Print the df
    print(parent_df)

    # Save transposed dataframe
    print('Writing to', args.write_path)
    parent_df.to_csv(args.write_path, index=True)
