"""Script for testing RF and MLR models (those that don't use tf data)."""

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import sys
import os
import pytz
import datetime
import yaml
import numpy as np
import pandas as pd
from collections import defaultdict
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError


src = os.path.join(os.getcwd(), '..')
if src not in sys.path:
    sys.path.append(src)


# Relative imports
from cli import test_others_cli  # nopep8
from models.mars_rf import build_mars_rf  # nopep8
from models.mars_mlr import build_mars_mlr  # nopep8
from models.dataset import Dataset  # nopep8
import utils  # nopep8


def rescale(data: np.ndarray, scaler: MinMaxScaler):
    """Rescaling 3D time series data.

    Predicted data will be of shape
    """
    if not isinstance(data, np.ndarray):
        try:
            data = data.numpy()  # works for tf tensors
        except:
            raise

    for t in range(data.shape[1]):
        data[:, t, :] = scaler.inverse_transform(data[:, t, :])
    return data


def write_testing(path: str, results: dict) -> None:
    """Writes the testing result to csv.

    :param yml_dict: <class 'dict'> that will be written as
        a yml file.

    :return:
    """

    # Raw lists
    if not isinstance(list(results.values())[0], list):
        df_results = pd.DataFrame(results, index=[0])
    else:
        df_results = pd.DataFrame(results)

    df_results.to_csv(path, index=False)

    return


if __name__ == '__main__':

    # CLI
    parser = test_others_cli('script for testing rf and mlr')
    args = parser.parse_args()
    args = utils.cast_args_to_bool(args)

    # Logging
    utc_begin = pytz.utc.localize(datetime.datetime.utcnow())
    cst_begin = utc_begin.astimezone(pytz.timezone("US/Central"))
    dtime = cst_begin.strftime('%Y%m%d_%H-%M-%S') + 'CST'
    log = utils.setup_logging(os.path.join(args.exp_path, f'{dtime}_info.log'))

    log.info(dtime)
    log.info(str(vars(args)))

    # Determine task
    if args.model == 'rf':

        with open(args.yml_path, 'r') as fobj:
            hparams = yaml.safe_load(fobj)['best_parameters']

        model = build_mars_rf(**hparams['model_params'], verbose=1)

        args.data_path = hparams['positional arguments']['data_path']

        log.info(args.data_path)

        if args.test_size is None:
            args.test_size = hparams['cv_params']['n_splits']

    elif args.model == 'mlr':
        model = build_mars_mlr(verbose=1)

    # Load data set
    data = Dataset(data_path=args.data_path)

    # Uses combined train-validation sets only for CV
    # NOTE: Data flattened since models expect single 2D input
    if args.which_dataset == 'validation':
        X = data.x_train_val.reshape(data.x_train_val.shape[0], -1)
        Y = data.y_train_val.reshape(data.x_train_val.shape[0], -1)

    # Uses combined train-validation-test set for CV...
    # this is for final model evaluation only
    elif args.which_dataset == 'test':
        X = np.concatenate(
            (data.x_train_val, data.x_test), axis=0).reshape(
                -1, data.x_train_val.shape[1] * data.x_train_val.shape[2])
        Y = np.concatenate(
            (data.y_train_val, data.y_test), axis=0).reshape(
                -1, data.x_train_val.shape[1] * data.x_train_val.shape[2])

    # Cross validate
    # Track scores for each fold
    scaled_scores = defaultdict(list)

    # Time series cross validator
    cv = TimeSeriesSplit(
        n_splits=args.n_splits,
        test_size=args.test_size)

    # Performance metrics
    # TODO: Parametrize
    metrics = [MeanAbsoluteError(), MeanSquaredError(),
               RootMeanSquaredError()]

    # Cross validation
    log.info('Cross validating...')
    for train_indices, validation_indices in cv.split(X):

        # Split indices for dataset...
        x_train, x_val = X[train_indices], X[validation_indices]
        y_train, y_val = Y[train_indices], Y[validation_indices]

        log.info('Fitting...')
        model.fit(x_train, y_train)

        # Get predictions
        y_val_preds = model.predict(x_val)

        # Reshape predictions and targets to 2D
        shape = data.y_train.shape[1:]
        log.info(shape)
        preds = y_val_preds.reshape(-1, *shape)
        targets = y_val.reshape(-1, *shape)

        # Rescale
        if args.rescale:
            log.info('Rescaling...')
            preds = rescale(
                data=preds, scaler=data.y_scaler)
            targets = rescale(data=targets, scaler=data.y_scaler)

        # Compute time related metrics
        for t in range(y_val_preds.shape[1]):
            for metric in metrics:
                scaled_scores[f'{metric.__class__.__name__}_{t+1}'].append(
                    metric(targets[:, t, :], preds[:, t, :]).numpy())

        # Flatten and check overall predictive capabilities
        for metric in metrics:
            scaled_scores[f'{metric.__class__.__name__}_overall'].append(metric(
                targets, preds).numpy())

    # Compute summary statistics for metrics
    summary_scaled_scores = {}
    for metric_key, metric_fold_list in scaled_scores.items():
        summary_scaled_scores[f'{metric_key}_mean'] = np.mean(
            metric_fold_list)
        summary_scaled_scores[f'{metric_key}_std'] = np.std(
            metric_fold_list)

    # Write the raw data
    raw_data_fname = dtime + \
        f'_scaled_{args.rescale}_raw_fold_results-{args.model}.csv'
    raw_data_path = os.path.join(args.exp_path, raw_data_fname)
    write_testing(path=raw_data_path, results=scaled_scores)

    # write summary data
    summary_data_fname = dtime + \
        f'_scaled_{args.rescale}_summary_fold_results-{args.model}.csv'
    summary_data_path = os.path.join(args.exp_path, summary_data_fname)
    write_testing(path=summary_data_path,
                  results=summary_scaled_scores)

    # Task complete
    log.info('\nEnd task!!')
