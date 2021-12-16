"""Module for tuning mars multioutput random forest regressor."""

import sklearn.pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
from keras_tuner import Objective
from keras_tuner.oracles import BayesianOptimizationOracle
from keras_tuner.tuners import SklearnTuner
from copy import deepcopy
import pickle
from collections import defaultdict
import sys
import os

# Time
import pytz
import datetime
import time

import numpy as np

import yaml
from yaml.representer import Representer
yaml.add_representer(defaultdict, Representer.represent_dict)


# Setup relative imports
src = os.path.join(os.getcwd(), '..')
if src not in sys.path:
    sys.path.append(src)

from models.mars_hypermodels import MultiOutputRFRHypermodel  # nopep8
from cli import tune_mars_rf_cli  # nopep8
import utils  # nopep8


if __name__ == '__main__':

    # CLI
    parser = tune_mars_rf_cli('tuning multioutput random forest model')
    args = parser.parse_args()

    # Track datetime CST
    utc_begin = pytz.utc.localize(datetime.datetime.utcnow())
    cst_begin = utc_begin.astimezone(pytz.timezone("US/Central"))
    dtime = cst_begin.strftime('%Y%m%d_%H-%M-%S') + 'CST'

    # Logging
    log = utils.setup_logging(os.path.join(
        args.exp_path, f'{dtime}_info.log'))

    # Begin log with datetime
    log.info(dtime)

    # Get argument groups
    arg_groups_dict = utils.get_arg_groups(args, parser)

    # Load data
    log.info('\nLoading data...')
    with open(args.data_path, 'rb') as fobj:
        x_train, x_val, x_test, y_train, y_val, y_test, _, _ = pickle.load(
            fobj).values()

    # Concatenate the data
    x_train_val = np.concatenate((x_train, x_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    # Flatten along last dimension (n, t * features)
    x_train_val = x_train_val.reshape(x_train_val.shape[0], -1)
    y_train_val = y_train_val.reshape(y_train_val.shape[0], -1)

    # Data dims
    log.info(
        f'x_train_val: {x_train_val.shape} - y_train_val: {y_train_val.shape}')

    # Instantiate the model -- deep copy to avoid mutating dict in main
    hypermodel = MultiOutputRFRHypermodel(deepcopy(arg_groups_dict))

    # Pre-tuner args
    log.info('\nInstantiating scoring metric and cross validation strat...')

    # Note: this is actually negative mean squared error, so it
    # is maximized
    log.info('\nInstantiating scoring metric...')
    scoring = metrics.make_scorer(metrics.mean_squared_error)

    # Used for sequentially related data (i.e., autocorrelated)
    log.info('\nInstantiating sklearn cross-validator...')
    cv = TimeSeriesSplit(n_splits=args.n_splits, test_size=args.test_size)

    # Tuner
    log.info('\nInstantiating tuner...')

    tuner = SklearnTuner(
        oracle=BayesianOptimizationOracle(
            objective=Objective('score', 'max'),
            max_trials=args.max_trials),
        hypermodel=hypermodel,
        scoring=scoring,
        cv=cv,
        directory=args.exp_path,
        project_name=dtime)

    # Time start
    start = time.time()

    # Search for best hyperparameters
    log.info('\nHyperparameter search...')
    tuner.search(x_train_val, y_train_val)

    # Time stop
    stop = time.time()

    # Difference in time
    time_dif = stop - start

    # Get hyperparameters
    log.info('\nRetrieving best hyperparameters...')
    tuner_best_hyperparameters = tuner.get_best_hyperparameters()[0].__dict__[
        'values']

    # Best val loss
    min_trial_metrics = utils.get_best_trial_metric(
        os.path.join(args.exp_path, dtime), 'score')

    # Abstract the best model process
    best_parameters = utils.get_best_parameters(
        arg_groups_dict=arg_groups_dict, tuner_best_hyperparameters=tuner_best_hyperparameters)

    # Update the groups dictionary
    arg_groups_dict['best_parameters'] = best_parameters
    arg_groups_dict['tuner_best_hyperparameters'] = tuner_best_hyperparameters
    arg_groups_dict['time_elapsed'] = f'{(time_dif)//(60*60)} hr {(time_dif//60)} min {int(float("." + str((time_dif)/60).split(".")[-1])*60)} s'
    arg_groups_dict['best_metric'] = min_trial_metrics

    # Dump the dictionary to yaml
    log_fname = f'{dtime}_mars_rf.yml'
    with open(os.path.join(args.exp_path, log_fname), 'w') as fobj:
        yaml.dump(arg_groups_dict, fobj)

    # Log completion
    log.info('\nTune mars rf complete.')
