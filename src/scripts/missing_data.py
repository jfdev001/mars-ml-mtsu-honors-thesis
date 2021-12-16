"""Script to find the percentage of sols for which all columns are nan."""

import os
import sys
import argparse
import pandas as pd
if not os.path.join(os.getcwd(), '..') in sys.path:
    sys.path.append(os.path.join(os.getcwd(), '..'))

from feature_engineering.sol_to_datetime import sol_to_datetime

if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser(
        'script for getting missing data information.')
    parser.add_argument(
        'data_path',
        help='path to csv file with data.',
        type=str)
    parser.add_argument(
        '--write_path',
        help='path to optional log file. (default: None)',
        type=str,
        default=None)
    parser.add_argument(
        '--time_column',
        help='name of column for tracking time in system. (default: SOL)',
        type=str,
        default='SOL')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data_path)
    try:
        df = df.drop('Unnamed: 0', axis=1)
    except:
        pass

    # Get the time series column as a series
    time_series = df[args.time_column]

    # Convert the time series column to datetime list
    dtime_lst = sol_to_datetime(time_series)

    # Set the dataframe index and drop the old time series column
    df = df.set_index(pd.to_datetime(dtime_lst))
    df = df.drop(labels=[args.time_column], axis=1)

    # Fill in df -- uses day for filling
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.asfreq.html
    df = df.asfreq('D')

    # Compute the number of columns in the df
    num_columns = len(df.columns)

    # Find the number of nan in a given row
    # https://stackoverflow.com/questions/30059260/python-pandas-counting-the-number-of-missing-nan-in-each-row
    nan_series = df.isnull().sum(axis=1)

    # Keep only those rows that have a null value for all values
    # of the column (i.e., every value is a nan and therefore)
    # the row must be completely imputed.
    missing_series = nan_series[nan_series == num_columns]

    # The report string
    report = f'For {args.data_path}:\n'
    report += f'Variables ({len(df.columns)}): {df.columns}\n'
    report += f'Number of missing values: {len(missing_series)}\n'
    report += f'Total number of rows including missing rows: {len(df)}\n'
    report += f'Percent of rows that are missing {round(len(missing_series)/len(df) * 100, 3)}%\n'

    # Save or not
    if args.write_path is not None:
        print(f'Saving report to {args.write_path}...')
        with open(args.write_path, 'w') as fobj:
            fobj.write(report)

    # Output report
    print(report)
