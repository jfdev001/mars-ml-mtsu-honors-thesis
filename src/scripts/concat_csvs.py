"""Script for concatenating data from various files into one csv.


```python
# Open interpreter by typing python in src/scripts 
 import os
 cwd = os.getcwd()
 unscaled = os.path.join(cwd, '..', '..', 'logs', 'testing', 'unscaled')

# Get paths
 all_file_paths = [os.path.join(unscaled, file) for file in os.listdir(unscaled)]
 raw_paths = [path for path in all_file_paths if path.find('raw') != -1]
 summary_paths = [path for path in all_file_paths if path.find('summary') != -1]
 early_stopping_paths = [path for path in all_file_paths if path.find('early') != -1]

# Get space separated paths
str_raw_paths = ' '.join(raw_paths)
str_summary_paths = ' '.join(summary_paths)
str_early_stopping_paths = ' '.join(early_stopping_paths)

# Concatenate
os.system(f'python concat_csvs.py {str_raw_paths} ../../logs/report/20211202_all_models_fold_mae_overall.csv --col_substr MeanAbsoluteError_overall')
os.system(f'python concat_csvs.py {str_raw_paths} ../../logs/report/20211202_all_models_fold_mse_overall.csv --col_substr MeanSquaredError_overall')
os.system(f'python concat_csvs.py {str_raw_paths} ../../logs/report/20211202_all_models_fold_rmse_overall.csv --col_substr RootMeanSquaredError_overall')

os.system(f'python concat_csvs.py {str_summary_paths} ../../logs/report/20211202_all_models_summary_mae_mean_overall.csv --col_substr MeanAbsoluteError_overall_mean')
os.system(f'python concat_csvs.py {str_summary_paths} ../../logs/report/20211202_all_models_summary_mse_mean_overall.csv --col_substr MeanSquaredError_overall_mean')
os.system(f'python concat_csvs.py {str_summary_paths} ../../logs/report/20211202_all_models_summary_rmse_mean_overall.csv --col_substr RootMeanSquaredError_overall_mean')

os.system(f'python concat_csvs.py {str_early_stopping_paths} ../../logs/report/20211202_all_models_num_epochs.csv --col_substr num_epochs')
```


"""

import argparse
from pathlib import Path
import os

import pandas as pd


def cli(description: str):

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        'read_paths',
        help='list of raw data files to combine into a single csv.',
        type=str,
        nargs='+')

    parser.add_argument(
        'write_path',
        help='path to write concatenated df as csv to.',
        type=str,)

    parser.add_argument(
        '--col_substr',
        help='column substring that a csv must contain for concatenation \
            to occur. By default, the script concatenates all columns \
            for all listed files. An example of a desired col_substr might be \
            `MeanSquaredError_overall`. The script would search through all the \
            read paths and extract only those columns from the each csv that possessed the `col_substr`.\
            Default to concatenating all columns in each read path.\
            (default: None)',
        type=str,
        default=None)

    return parser


def main():

    # CLI
    parser = cli('concatenate certain columns in various csvs to one csv')
    args = parser.parse_args()

    # Init the parent dataframe
    parent_df = pd.DataFrame()

    # Iterate through paths and create dataframe
    print(
        f'Iterating through raw data paths and extracting substr = `{args.col_substr}`')
    for path in args.read_paths:

        # Load the dataframe and extract columns
        loaded_df = pd.read_csv(path)
        columns = loaded_df.columns

        # Paths will always have '-' followed by the model name
        # and will also always end in '.csv'... splitting and indexing
        # the appropriate elements extracts the model name
        model_name = Path(path).name.split('-')[-1].split('.')[0]

        # Rename the variables based on the model name to prevent
        # collisions when merging...
        rename_dict = {
            col: col + f'-{model_name}' for col in columns}

        # Rename loaded df columns
        loaded_df = loaded_df.rename(columns=rename_dict)

        # New columns that resulted from renaming
        renamed_columns = loaded_df.columns

        # Determine substring extraction
        if args.col_substr is not None:

            # TODO: Fix with regex
            # Special case of rmse or mse
            if args.col_substr.find('MeanSquaredError') != -1 \
                    and args.col_substr.find('Root') == -1:

                print('Special case...')

                # Extract columns based on substring (and avoid rmse)
                selected_columns = [
                    col for col in renamed_columns
                    if (col.find(args.col_substr) != -1
                        and col.find('RootMeanSquaredError') == -1)]

            else:

                print('Normal case...')

                # Extract columns based on substring
                selected_columns = [
                    col for col in renamed_columns
                    if col.find(args.col_substr) != -1]

            # Load only the dataframe matching the selected columns
            loaded_df = loaded_df[selected_columns]

        # Determine append or join
        if len(parent_df) == 0:
            parent_df = parent_df.append(loaded_df)
        else:
            parent_df = parent_df.join(loaded_df)

    # Logging number of columns in csv
    print('Num columns in outfile:', len(parent_df.columns))

    # Validate write path
    if not args.write_path.endswith('.csv'):
        print('Appending .csv to path...')
        args.write_path += '.csv'

    # Write result to file
    print(f'Writing to {args.write_path}')
    parent_df.to_csv(args.write_path, index=False)


if __name__ == '__main__':
    main()
