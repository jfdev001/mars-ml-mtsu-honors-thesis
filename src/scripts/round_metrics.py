"""Script that can round all values in a csv by the desired value."""

import pandas as pd
import argparse
from pathlib import Path
import os

if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser('round data in csv')
    parser.add_argument(
        'csv_path',
        help='path to csv file whose values will be rounded.',
        type=str)
    parser.add_argument(
        'round',
        help='number of digits to include in the rounding.',
        type=int)
    parser.add_argument(
        '--write_path',
        help='path to write the new csv. (default: Name of CSV with round_n appended).',
        default=None)
    args = parser.parse_args()

    # Set default write path
    if args.write_path is None:
        path = Path(args.csv_path)
        args.write_path = os.path.join(
            path.parent, path.name.strip('.csv') + f'_round_{args.round}.csv')

    # Load dataframe
    print('\nLoading data:', args.csv_path)
    df = pd.read_csv(args.csv_path)

    # Round it
    print('\nRounding:', args.round)
    df = df.round(decimals=args.round)

    # Write it
    print('\nWriting to file:', args.write_path)
    df.to_csv(args.write_path, index=False)
