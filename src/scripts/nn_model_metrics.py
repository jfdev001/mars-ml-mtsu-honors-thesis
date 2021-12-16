"""Module to get metrics from validation set of neural nets.

TODO: Deprecated... summary and raw results are from NNTester cross
validation.
"""

import datetime
import pytz
import time
from pathlib import Path
import numpy as np
import os
import sys
import pandas as pd
import yaml
from yaml.representer import Representer
from collections import defaultdict
yaml.add_representer(defaultdict, Representer.represent_dict)

# Setup relative imports
src = os.path.join(os.getcwd(), '..')
if src not in sys.path:
    sys.path.append(src)

# Relative imports
from cli import nn_model_metrics_cli  # nopep8


if __name__ == '__main__':
    # CLI
    parser = nn_model_metrics_cli(
        description='extract model metrics for neural nets for report')
    args = parser.parse_args()

    # Track datetime CST
    utc_begin = pytz.utc.localize(datetime.datetime.utcnow())

    cst_begin = utc_begin.astimezone(pytz.timezone("US/Central"))

    dtime = cst_begin.strftime('%Y%m%d_%H-%M-%S') + 'CST'

    # Extract yml file names
    ymls = [os.path.join(args.training_path, f)
            for f in os.listdir(args.training_path) if f.endswith('yml')]

    # Iterate through the files and get the appropriate information...
    # mean & std...
    report = defaultdict(dict)
    report['ymls'] = ymls

    print('\nIterating through yaml and extracting mean/std of metrics...')

    # Path to yaml file with history of metrics and unique id
    for model_num, y_path in enumerate(ymls):

        # Load the yaml file as a dictionary
        with open(y_path, 'r') as fobj:
            metric_data_dict = yaml.safe_load(fobj)

        # The parent key is the path associated with a particular
        # yaml and will be used
        parent_key = Path(y_path).name

        # For a particular yaml dictionary, there will be a unique
        # identifier for a given neural net
        try:
            report[parent_key]['special_id'] = metric_data_dict['special_id']
        except KeyError:
            report[parent_key]['special_id'] = f'model_{model_num}'

        # Iterate through key, values in the data dictionary
        for k, v in metric_data_dict.items():

            # Determine if key is validation only because training
            # metrics should not be used to evaluate the performance
            # of the model
            if 'val' in k and 'loss' not in k:
                mean_val = float(np.mean(v))
                std_val = float(np.std(v))

                # Populate the report dictionary by adding the mean
                # and standard deviation of a given metric to the key
                report[parent_key][f'{k}_mean'] = mean_val
                report[parent_key][f'{k}_std'] = std_val

    # Process the report dictionary into a pd for csv
    df_dict = defaultdict(list)

    for k, v_dict in report.items():

        # Prevents non-history dictionaries from being processed
        if 'history' in k:

            # Use the special id as the column key
            special_id_name = v_dict['special_id']

            # Child values as a list
            child_values = list(v_dict.values())

            # Iterate through child values...
            # appending to the default dict as long as the value
            # is not a string
            for v in child_values:
                if not isinstance(v, str):
                    df_dict[special_id_name].append(v)

    # Values if dictionary
    report_values = [v for v in report.values() if isinstance(v, dict)]

    # Get names for indices
    ix_names = [metric.lstrip(
        'val_') for metric in report_values[0].keys() if metric.find('val') != -1]

    # Make dataframe
    df = pd.DataFrame(df_dict)

    # Change the indices to reflect the metric association
    df.index = ix_names

    # Save the dataframe as a csv
    print('\nSaving metric reports as csv...')
    write_path = os.path.join(
        args.report_path, f'{dtime}_{Path(y_path).parent.name}.csv')
    df.to_csv(write_path)

    # # Save the report dictionary
    # print('\nSaving metric reports...')
    # with open(os.path.join(args.report_path, f'{dtime}_{Path(y_path).parent.name}.yml'), 'w') as fobj:
    #     yaml.dump(report, fobj)
