"""Module to report the best models for each model from tuning.

Save best hyperparameters yml in it's folder and sort the models
in their respective directories in descending order of val loss.
"""
import sys
from pathlib import Path
import os

from distutils.util import strtobool

import yaml

import numpy as np

# Setup relative imports
src = os.path.join(os.getcwd(), '..')
if src not in sys.path:
    sys.path.append(src)

# Relative imports
from cli import best_models_cli  # nopep8

if __name__ == '__main__':

    # CLI parse
    parser = best_models_cli('get best model hyperparameters')
    args = parser.parse_args()

    # Validate read path
    if not os.path.exists(args.yml_dir):
        raise OSError(f'{args.yml_dir} does not exist.')

    # Create out dir if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # Overwrite out dir if desired
    elif (os.path.exists(args.out_dir) and bool(strtobool(args.overwrite_out_dir))):
        for f in os.listdir(args.out_dir):
            os.remove(os.path.join(args.out_dir, f))

    # Parallel lists that will be used to extract the best
    # hyperparameters
    val_losses = []
    ymls = [f for f in os.listdir(args.yml_dir) if f.endswith('yml')]

    # Iterate through yml files in the list of yml files
    for y in ymls:

        # Open and load yaml data
        with open(os.path.join(args.yml_dir, y), 'r') as fobj:
            data = yaml.safe_load(fobj)

        # Attempt to get the relevant keys and values from the yml
        # if unable, put placeholder value in val losses to represent
        # the file
        try:

            # Extract the desired metric
            best_metric = data['best_metric']

            # Special identifier supplied
            if args.special_id is not None:

                # Determine matching identifier
                if data['optional arguments']['special_id'] == args.special_id:
                    val_losses.append(best_metric)
                else:
                    val_losses.append(np.nan)

            # No special identifier
            else:
                val_losses.append(best_metric)

        except:
            val_losses.append(np.nan)

    # Get the index in the yaml files list corresponding
    # to the lowest val loss (excluding the nan)
    best_yml = ymls[val_losses.index(np.nanmin(val_losses))]

    # Logging
    print('The best yml file is: ', best_yml)

    # Get the best data
    with open(os.path.join(args.yml_dir, best_yml), 'r') as fobj:
        best_data = yaml.safe_load(fobj)

    # Save the best data to the out dir
    with open(os.path.join(args.out_dir, best_yml), 'w') as fobj:
        yaml.dump(best_data, fobj)

    # Logging
    print('Best model yml has been saved to ', os.path.join(
        args.out_dir, best_yml), '\nScript complete!')
