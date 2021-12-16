"""Module for training and validating multiple linear regression."""

import sys
import os
import yaml
import pickle
from joblib import dump, load
from collections import defaultdict
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

import pytz
import datetime

# Setup relative imports
src = os.path.join(os.getcwd(), '..')
if src not in sys.path:
    sys.path.append(src)

# Relative imports
from cli import train_mars_mlr_cli  # nopep8
import utils  # nopep8
from models.mars_mlr import build_mars_mlr  # nopep8


if __name__ == '__main__':

    # CLI for data only
    parser = train_mars_mlr_cli('train multioutput mlr model.')
    args = parser.parse_args()

    # Track datetime CST
    utc_begin = pytz.utc.localize(datetime.datetime.utcnow())

    cst_begin = utc_begin.astimezone(pytz.timezone("US/Central"))

    dtime = cst_begin.strftime('%Y%m%d_%H-%M-%S') + 'CST'

    # Set up logger
    log = utils.setup_logging(os.path.join(args.exp_path, f'{dtime}_info.log'))

    # Begin log with datetime
    log.info(dtime)

    # Load data
    log.info(f'\nLoading and partitioning data: {args.data_path}')

    with open(args.data_path, 'rb') as fobj:
        x_train, x_val, x_test, y_train, y_val, y_test, scaler = pickle.load(
            fobj).values()

    # Check dims of data
    if len(x_train.shape) > 2 or len(y_train.shape) > 2:
        log.info('\nReshaping data...')
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
        y_train = y_train.reshape(y_train.shape[0], -1)
        y_val = y_val.reshape(y_val.shape[0], -1)

     # Data dims
    log.info(f'x_train: {x_train.shape} - y_train: {y_train.shape}')

    log.info(f'x_val: {x_val.shape} - y_val: {y_val.shape}')

    # Instantiate model
    log.info('\nInstantiating model...')
    model = build_mars_mlr()

    # Fit the model
    log.info('\nTraining model...')
    model.fit(x_train, y_train)

    # Save model
    log.info('\nSave model...')
    dump(model, os.path.join(args.exp_path, f'{dtime}_rf.joblib'))

    # Make predictions
    log.info('\nMake predictions...')
    y_val_pred = model.predict(x_val)

    # Log shape
    log.info(f'y_val_pred: {y_val_pred.shape}')

    # Dividing validation x val and y val into equal indices
    # and computing metrics for each fold
    log.info('\nGet folds of size 32 and compute metrics, then avg and std...')
    report = defaultdict(list)
    for i in range(0, x_val.shape[0] - 32, 32):

        # Scoring
        mae = mean_absolute_error(y_val[i:i + 32], y_val_pred[i:i + 32])
        mse = mean_squared_error(y_val[i:i + 32], y_val_pred[i:i + 32])
        rmse = mse ** (1/2)

        # Append to dictionary
        report['mae'].append(mae)
        report['mse'].append(mse)
        report['rmse'].append(rmse)

    # Get averages
    stats = {}
    for k, v in report.items():
        stats[f'{k}_mean'] = float(np.mean(v))
        stats[f'{k}_std'] = float(np.std(v))

    # rf metrics
    stats['data_path'] = args.data_path
    log.info('\nSaving metrics')
    with open(os.path.join(args.exp_path, f'{dtime}_MLR_metrics.yml'), 'w') as fobj:
        yaml.dump(stats, fobj)

    # Log exit
    log.info('\nMLR training and evaluation complete!!')
