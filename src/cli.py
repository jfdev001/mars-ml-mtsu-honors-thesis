"""Module for argument parser function."""

import argparse
import os
from pathlib import Path

from yaml import parse

# TODO: Remove global dirs
# TODO: Add logging
# TODO: Remove defaults
# TODO: All args for hyperparameter tuning should be nargs

ROOT_DIR = Path(os.getcwd()).resolve().parents[1]
DATA_DIR = os.path.join(ROOT_DIR, 'data')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')


def data_cli(description):
    """Returns CLI object for data processing.

    Should have tasks all separate so that a single task is
    completed per need??

    :param description: <class 'str'>
    """

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        'task',
        choices=[
            'clean_split_impute',
            'scale_rank3',
            'pca',
        ],
        help='name of the task (clean_split_impute: clean, train-val-test split, then impute data; \
                                scale_rank3: rank-3 (timeseries) formatted, scaled xy data tensors; \
                                pca: feature reduction of rank-3 (timeseries) formatted, scaled xy data tensors')

    # Arg groups
    path_params = parser.add_argument_group(
        'path_params',
        'params for different paths relative to the scripts/ directory.')

    clean_params = parser.add_argument_group(
        'clean_params',
        'params for cleaning data.')

    split_params = parser.add_argument_group(
        'split_params',
        'params for splitting data into training, validation, and testing \
            (holdout validation) groups.')

    impute_params = parser.add_argument_group(
        'impute_params',
        'params for missing value imputation.')

    scale_params = parser.add_argument_group(
        'scale_params',
        'params for scaling data.')

    rank3_params = parser.add_argument_group(
        'rank3_params',
        'params for windowing time series data.')

    pca_params = parser.add_argument_group(
        'pca_params',
        'params for feature reduction via PCA.')

    # Pathing
    path_params.add_argument(
        '--unclean_data_path',
        help='path to csv format data with feature/target labels as headers, \
            a SOL column, and not timeseries formmatted.',
        type=str)

    path_params.add_argument(
        '--data_path',
        help=f'path to data that has been imputed and split to train, validation \
        and testing sets.',
        type=str)

    path_params.add_argument(
        '--exp_path',
        help=f'path to experiment log files (arguments to script).',
        type=str,
        required=True)

    # Cleaning
    clean_params.add_argument(
        '--nan_threshold',
        help='variables with MORE than :param nan_threshold: are dropped.',
        type=int,
        default=1300)

    # Splitting
    split_params.add_argument(
        '--test_ratio',
        help='the proportion of the dataset for testing. (default: 0.2)',
        type=float,
        default=0.2)

    split_params.add_argument(
        '--val_ratio',
        help='the proportion of the dataset for validation (default: 0.2)',
        type=float,
        default=0.2)

    split_params.add_argument(
        '--skip_verification',
        choices=['True', 'False'],
        help='True to skip user verification of splitting sizes, False \
            otherwise. (default: False)',
        type=str,
        default='False')

    # Imputation
    impute_params.add_argument(
        '--imputer',
        choices=[
            'bayesian_ridge',
            'decision_tree',
            'extra_trees',
            'knn'],
        help='the estimator used for iterative imputation. (default: extra_trees)',
        type=str,
        default='extra_trees')

    impute_params.add_argument(
        '--n_jobs',
        help='specifies the max number of concurrent workers for \
            :param extra_trees: and :param knn: in iterative imputation. (default: -2)',
        type=int,
        default=-2)

    impute_params.add_argument(
        '--n_estimators',
        help='number of decision trees in the extra trees forest in iterative \
            imputation. (default: 100)',
        type=int,
        default=100)

    impute_params.add_argument(
        '--n_neighbors',
        help='number of neighbors to use for kneighbors queries in iterative \
            imputation. (default: 5)',
        type=int,
        default=5)

    impute_params.add_argument(
        '--tree_random_state',
        help='controls three sources of randomness for both decision and extra \
            trees in iterative imputation. (default: 42)',
        type=int,
        default=42)

    impute_params.add_argument(
        '--iter_impute_random_state',
        help='control <class `sklearn.impute.IterativeImputer`> random state. (default: 42)',
        type=int,
        default=42)

    impute_params.add_argument(
        '--max_impute_iter',
        help='maximum number of iteration for \
            <class `sklearn.impute.IterativeImputer`>. (default: 10)',
        type=int,
        default=10)

    # Feature reduction

    # parser.add_argument(
    #     '--backward_elimination_alpha',
    #     help='significance level for eliminating features via backward \
    #         elimination (BE) for MLR. BE not used by default. (default: -1.0)',
    #     type=float,
    #     default=-1.0,)

    pca_params.add_argument(
        '--prop_explained_var',
        help='target explained var for PCA. Data MUST be standardized. \
            (default: 0.95)',
        type=float,
        default=-1.0)

    # Scale
    scale_params.add_argument(
        '--scaler',
        choices=['std_scaler', 'minmax_scaler', 'no_scaler'],
        help='the scaling routine to apply to data. (default: minmax_scaler).',
        default='minmax_scaler')

    # Timestep formatting
    rank3_params.add_argument(
        '--labels',
        help='comma separated list (no space after comma unless it`s part of \
            label name) of labels (aka targets). MUST use `" "` around args if \
            the arg contains symbols such as `(`. \
            (default: Mean Ambient Air Temperature (K))',
        type=str,
        default='Mean Ambient Air Temperature (K)')

    rank3_params.add_argument(
        '--x_timesteps',
        help='number of input timesteps (previous days) used to predict output \
        (future days). (default: 3)',
        type=int,
        default=3)

    rank3_params.add_argument(
        '--y_timesteps',
        help='number of output timesteps (future days) to predict. MUST be \
            negative. (default: -7)',
        type=int,
        default=-7)

    rank3_params.add_argument(
        '--overlapping_windows',
        choices=['True', 'False'],
        help='True for overlapping timeseries windows, False otherwise. \
            (default: True)',
        default='True')

    return parser


def tune_simple_mars_nn_cli(description):
    """Returns CLI object for tuning simple mars neural network.

    Arguments are informed by <class 'SimpleMarsNN'> defined in 
    'mars_nn.py'

    :param description: <class 'str'>
    """

    # CLI Object
    parser = argparse.ArgumentParser(description=description)

    # Argument groups
    model_params = parser.add_argument_group(
        'model_params',
        'parameters for model specific instantiation.')

    fit_params = parser.add_argument_group(
        'fit_params',
        'parameters used during model.fit().')

    compile_params = parser.add_argument_group(
        'compile_params',
        'parameters for model compilation (learning rate, etc.).')

    oracle_params = parser.add_argument_group(
        'oracle_params',
        'parameters related to the keras-tuner Tuner base class for \
            the oracle search algorithm.')

    cv_params = parser.add_argument_group(
        'cv_params',
        'parameters for cross validation strategy.')

    # Data and logging paths
    parser.add_argument(
        'data_path',
        help=f'path to data that has been imputed and split to train, validation \
        and testing sets with features/targets')

    parser.add_argument(
        '--exp_path',
        help=f'path to log files folder (model checkpoints, tensorboard logs, \
            model outputs).',
        required=True,
        type=str)

    # Misc
    parser.add_argument(
        '--search_strategy',
        choices=['cv_bopt', 'bopt'],
        help='str for tuner search strategy \
            (cv_bopt: cross validation bayesian optimization; \
            bopt: bayesian optimization). (default: cv_bopt)',
        default='cv_bopt')

    parser.add_argument(
        '--special_id',
        help='special identifier to be appended to log files if desired. \
            (default: None)',
        type=str,
        default=None)

    # Model parameters
    model_params.add_argument(
        '--autoregressive',
        help='constant or nargs of str (True False) for choice list for \
            autoregressive model. (default: False)',
        nargs='+',
        type=str,
        default=['False'])

    model_params.add_argument(
        '--rnn_size',
        help='constant or nargs of int for choice list for rnn size. (default: 1)',
        type=int,
        nargs='+',
        default=[1])

    model_params.add_argument(
        '--rnn_layers',
        help='constant or nargs of int for choice list for rnn layers. (default: 1)',
        type=int,
        nargs='+',
        default=[1])

    model_params.add_argument(
        '--rnn_cell',
        help='constant or nargs of str for choice list for rnn cell. (default: gru)',
        type=str,
        nargs='+',
        default=['gru'])

    model_params.add_argument(
        '--dropout',
        help='constant float or nargs of float for dropout rate. (default: 0.0)',
        type=float,
        nargs='+',
        default=[0.0])

    # Fit parameters
    fit_params.add_argument(
        '--batch_size',
        help='batch size for training of each model in fold or execution trial.',
        type=int,
        required=True)

    fit_params.add_argument(
        '--drop_remainder',
        choices=['True', 'False'],
        help='True to drop remainder for batching, False otherwise. (default: True)',
        type=str,
        default='True')

    fit_params.add_argument(
        '--iid',
        help='independent and identically distributed. (default: True)',
        choices=['True', 'False'],
        type=str,
        default='True')

    fit_params.add_argument(
        '--shuffle_batched_seq',
        help='if :param iid: is False, then True for shuffling batched time \
            sequences series, False otherwise. (default: False)',
        choices=['True', 'False'],
        type=str,
        default='False')

    fit_params.add_argument(
        '--epochs',
        help='number of epochs for training of each model in fold or execution trial. (default: 2)',
        type=int,
        default=2)

    fit_params.add_argument(
        '--es_patience',
        help='patience for early stopping callback. (default: 1)',
        type=int,
        default=1)

    # Compile parameters
    compile_params.add_argument(
        '--learning_rate',
        help='constant float or nargs of float for Adam learning rate. (default: 0.001)',
        type=float,
        nargs='+',
        default=[0.001])

    # Oracle keras-tuner parameters
    oracle_params.add_argument(
        '--max_trials',
        help='max number of trials to run for hyperparameter search algorithm. (default: 1)',
        type=int,
        default=1)

    oracle_params.add_argument(
        '--executions_per_trial',
        help='only if :param search_strategy: is bopt, executions per trial to reduce variance for hyperparameter search. (default: 1)',
        type=int,
        default=1)

    # Cross validation parameters
    cv_params.add_argument(
        '--cv',
        choices=['TimeSeriesSplit', 'KFold'],
        help='name of sklearn cross-validator to use with :param cv_bopt:. (default: TimeSeriesSplit)',
        type=str,
        default='TimeSeriesSplit')

    cv_params.add_argument(
        '--n_splits',
        help='number of splits for cross validator with :param cv_bopt:. (default: 5)',
        type=int,
        default=5)

    # The modified CLI Object
    return parser


def tune_conv_mars_nn_cli(description):
    """CLI for tuning conv mars nn"""

    parser = argparse.ArgumentParser(description=description)

    # CLI Object
    parser = argparse.ArgumentParser(description=description)

    # Argument groups
    model_params = parser.add_argument_group(
        'model_params',
        'parameters for model specific instantiation.')

    fit_params = parser.add_argument_group(
        'fit_params',
        'parameters used during model.fit().')

    compile_params = parser.add_argument_group(
        'compile_params',
        'parameters for model compilation (learning rate, etc.).')

    oracle_params = parser.add_argument_group(
        'oracle_params',
        'parameters related to the keras-tuner Tuner base class for \
            the oracle search algorithm.')

    cv_params = parser.add_argument_group(
        'cv_params',
        'parameters for cross validation strategy.')

    # Data and logging paths
    parser.add_argument(
        'data_path',
        help=f'path to data that has been imputed and split to train, validation \
        and testing sets with features/targets')

    parser.add_argument(
        '--exp_path',
        help=f'path to log files folder (model checkpoints, tensorboard logs, \
            model outputs).',
        required=True,
        type=str)

    # Misc
    parser.add_argument(
        '--search_strategy',
        choices=['cv_bopt', 'bopt'],
        help='str for tuner search strategy \
            (cv_bopt: cross validation bayesian optimization; \
            bopt: bayesian optimization). (default: cv_bopt)',
        default='cv_bopt')

    parser.add_argument(
        '--special_id',
        help='special identifier to be appended to log files if desired. \
            (default: None)',
        type=str,
        default=None)

    # Model parameters
    model_params.add_argument(
        '--activation',
        help='constant or nargs of str for cnn activation function. (default: relu)',
        nargs='+',
        type=str,
        default=['relu'])

    model_params.add_argument(
        '--filters',
        help='constant or nargs of int for filters (default: 32)',
        type=int,
        nargs='+',
        default=[32])

    model_params.add_argument(
        '--kernel_size',
        help='constant or nargs of int (tuples of ints) for kernel (default: 3)',
        type=int,
        nargs='+',
        default=[3])

    model_params.add_argument(
        '--filter_increase_rate',
        help='constant or nargs of int for rate by which filters should increase \
            for subsequent layers. Has no effect if :param conv_layers: is 1. \
            (default: 1)',
        type=int,
        nargs='+',
        default=1)

    model_params.add_argument(
        '--conv_layers',
        help='constant or nargs of int for number of cnn layers (default: 1)',
        type=int,
        nargs='+',
        default=[1])

    model_params.add_argument(
        '--conv_padding',
        help='constant or nargs of str in [`valid`, `same`, `causal`] \
            for cnn padding. (default: valid)',
        type=str,
        nargs='+',
        default=['valid'])

    model_params.add_argument(
        '--dilation_rate',
        help="constant or nargs of int for the base of an exponent to the power \
            of the i^th convolutional layer by which the dilation \
            rate will increase for subsequent layers. Has no effect if \
            :param conv_layers: is 1. For example, if :param dilation_rate: \
            is 2 and :param conv_layers: is 3, the dilation rate for the 0th \
            layer will be 2**0, then for the 1th layer 2**1, then for the 2th \
            layer 2**2. (default: 1)",
        nargs='+',
        type=int,
        default=[1])

    model_params.add_argument(
        '--conv_strides',
        help='constant or nargs of int for cnn strides. (default: 1)',
        type=int,
        nargs='+',
        default=[1])

    model_params.add_argument(
        '--use_pooling',
        help='constant or nargs of str in [`True`, `False`] for using \
            pooling. (default: True)',
        type=str,
        nargs='+',
        default=['True'])

    model_params.add_argument(
        '--pool_size',
        help='constant or nargs of int for pool size. (default: 2)',
        type=int,
        nargs='+',
        default=[2])

    model_params.add_argument(
        '--pool_strides',
        help='constant or nargs of int for pool strides. (default: None)',
        type=int,
        nargs='+',
        default=[None])

    model_params.add_argument(
        '--pool_padding',
        help='constant or nargs of str in [`valid`, `same`]. \
            (default: valid)',
        type=str,
        nargs='+',
        default=['valid'])

    model_params.add_argument(
        '--dropout_rate',
        help='constant or nargs of float for dropout rate. \
            (default: 0.0)',
        type=float,
        nargs='+',
        default=[0.0])

    # Fit parameters
    fit_params.add_argument(
        '--batch_size',
        help='batch size for training of each model in fold or execution trial',
        type=int,
        required=True)

    fit_params.add_argument(
        '--drop_remainder',
        choices=['True', 'False'],
        help='True to drop remainder for batching, False otherwise. (default: True)',
        type=str,
        default='True')

    fit_params.add_argument(
        '--iid',
        help='independent and identically distributed. (default: True)',
        choices=['True', 'False'],
        type=str,
        default='True')

    fit_params.add_argument(
        '--shuffle_batched_seq',
        help='if :param iid: is False, then True for shuffling batched time \
            sequences series, False otherwise. (default: False)',
        choices=['True', 'False'],
        type=str,
        default='False')

    fit_params.add_argument(
        '--epochs',
        help='number of epochs for training of each model in fold or execution \
            trial. (default: 2)',
        type=int,
        default=2)

    fit_params.add_argument(
        '--es_patience',
        help='patience for early stopping callback. (default: 1)',
        type=int,
        default=1)

    # Compile parameters
    compile_params.add_argument(
        '--learning_rate',
        help='constant float or nargs of float for Adam learning rate. \
            (default: 0.001)',
        type=float,
        nargs='+',
        default=[0.001])

    # Oracle keras-tuner parameters
    oracle_params.add_argument(
        '--max_trials',
        help='max number of trials to run for hyperparameter search algorithm. \
            (default: 1)',
        type=int,
        default=1)

    oracle_params.add_argument(
        '--executions_per_trial',
        help='only if :param search_strategy: is bopt, executions per trial to \
            reduce variance for hyperparameter search. (default: 1)',
        type=int,
        default=1)

    # Cross validation parameters
    cv_params.add_argument(
        '--cv',
        choices=['TimeSeriesSplit', 'KFold'],
        help='name of sklearn cross-validator to use with :param cv_bopt:. (default: TimeSeriesSplit)',
        type=str,
        default='TimeSeriesSplit')

    cv_params.add_argument(
        '--n_splits',
        help='number of splits for cross validator with :param cv_bopt:. (default: 5)',
        type=int,
        default=5)

    # The modified CLI Object
    return parser


def train_nns_cli(description):
    """CLI for training Neural Nets."""

    parser = argparse.ArgumentParser(description=description)

    # Data and logging paths
    parser.add_argument(
        'yml_path',
        help='path to yml file containing the hyperparameters for training \
            (includes data path relative to scripts/ directory).',
        type=str,)

    parser.add_argument(
        '--exp_path',
        help=f'path to log files folder (model checkpoints, tensorboard logs, \
            model outputs).',
        required=True,
        type=str)

    return parser


def test_nns_cli(description):
    """CLI for testing neural nets."""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        'model',
        help='model to load.',
        choices=['conv_mars_nn', 'simple_mars_nn'])

    parser.add_argument(
        'yml_path',
        help='path to hyperparameter folder. \
            NOTE: If providing this arg, do not provide `--history_path` arg.',
        type=str)

    parser.add_argument(
        '--history_path',
        help='path to yml file in training folder that holds model history. \
            NOTE: DEPRECATED!!! If providing this arg, do not provide `--yml_path` arg.',
        type=str,
        default=None)

    parser.add_argument(
        '--task',
        choices=['cross_validate'],
        help='evaluation task. (default: cross_validate)',
        default='cross_validate')

    parser.add_argument(
        '--exp_path',
        help=f'path to log files folder (model checkpoints, tensorboard logs, \
            model outputs).',
        required=True,
        type=str)

    data_params = parser.add_argument_group(
        'data_params',
        'params for the data specification.')

    data_params.add_argument(
        '--which_dataset',
        choices=['validation', 'test'],
        help='for which dataset to test the trained model on. (default: validation)',
        default='validation')

    cv_params = parser.add_argument_group(
        'cv_params',
        'params for walkforward cross validation.')

    cv_params.add_argument(
        '--n_splits',
        help='splits for TimeSeriesSplit. (default: None)',
        type=int,
        default=None)

    cv_params.add_argument(
        '--rescale',
        choices=['True', 'False'],
        help='True to rescale outputs, False otherwise. (default: True)',
        default='True')

    return parser


def test_others_cli(description):
    """CLI for testing non-Neural net models."""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('model', choices=['mlr', 'rf'], help='model to use.')

    parser.add_argument(
        '--yml_path',
        help='path to rf hyperparameter yml. (required for RF but not MLR)',
        type=str)

    parser.add_argument(
        '--data_path',
        help='path to data. (required for MLR but not RF).',
        type=str)

    parser.add_argument(
        '--exp_path',
        help=f'path to log files folder (model checkpoints, tensorboard logs, \
            model outputs).',
        required=True,
        type=str)

    parser.add_argument(
        '--task', choices=['cross_validate'],
        help='testing task. (default: cross_validate)',
        default='cross_validate')

    data_params = parser.add_argument_group(
        'data_params',
        'params for data specification')

    data_params.add_argument(
        '--which_dataset',
        choices=['validation', 'test'],
        help='for which dataset to test the trained model on. (default: validation)',
        default='validation')

    cv_params = parser.add_argument_group(
        'cv_params',
        'params for cross validation.')

    cv_params.add_argument(
        '--n_splits',
        help='splits for cv. (default: 5)', 
        type=int,
        default=5)

    cv_params.add_argument(
        '--test_size',
        help='test size for time series cv. Required for MLR. (default None)',
        type=int,
        default=None)

    cv_params.add_argument(
        '--rescale',
        choices=['True', 'False'],
        help='True to rescale outputs, False otherwise. (default: True)',
        default='True')

    return parser


def tune_mars_rf_cli(description):
    """CLI for tuning multioutput random forest."""

    parser = argparse.ArgumentParser(description=description)

    # Pathing
    parser.add_argument(
        'data_path',
        help=f'path to data that has been imputed and split to train, validation \
            and testing sets with features/targets')

    parser.add_argument(
        '--exp_path',
        help=f'path to log files folder (model checkpoints, tensorboard logs, \
            model outputs).',
        required=True,
        type=str)

    # Cross validation args
    cross_val = parser.add_argument_group(
        'cv_params',
        'parameters for cross validation (time series).')

    cross_val.add_argument(
        '--n_splits',
        help='split for cross validation. (default: 5)',
        type=int,
        default=5)

    cross_val.add_argument(
        '--test_size', help='test size for cv. (default: 28)', type=int, default=28)

    # Oracle args
    oracle_params = parser.add_argument_group(
        'oracle_params',
        'parameters for keras-tuner search algorithm.')

    oracle_params.add_argument(
        '--max_trials',
        help='max number of trials to run for hyperparameter search algorithm. (default: 1)',
        type=int,
        default=1)

    # Model parameters
    model_params = parser.add_argument_group(
        'model_params',
        'parameters for model specific instantiation.')

    model_params.add_argument(
        '--n_rf_jobs',
        help='parallelize random forest regressor. (default: -1)',
        type=int,
        default=-1)

    model_params.add_argument(
        '--n_mo_jobs',
        help='parallelize multioutput regressor. (default: -1)',
        type=int,
        default=-1)

    model_params.add_argument(
        '--n_estimators',
        help='const or nargs of int for decision trees in forest. (default: 100)',
        type=int,
        nargs='+',
        default=[100])

    model_params.add_argument(
        '--max_depth',
        help='const or nargs of int for longest path between root and leaf node. \
            (default: None',
        type=int,
        nargs='+',
        default=[None])

    model_params.add_argument(
        '--min_samples_split',
        help='const or nargs of int for minimum required number of observations \
            to split a node. (default: 2)',
        type=int,
        nargs='+',
        default=[2])

    model_params.add_argument(
        '--min_samples_leaf',
        help='const or nargs of int minimum number of samples present in \
            leaf node after splitting it. (default: 1)',
        type=int,
        nargs='+',
        default=[1])

    return parser


def train_mars_rf_cli(description):
    """CLI for training the multiouput random forest regressor."""

    parser = argparse.ArgumentParser(description=description)

    # What are the things that i will need for the training of the CLI?
    # path to data
    # path to yml for model hyperparameters
    # path for model outputs (tensorboard, ckpt, etc.)

    # Data and logging paths

    parser.add_argument(
        'yml_path',
        help='path to yml file containing the hyperparameters for training \
            (includes data path relative to scripts/ directory).',
        type=str,)

    parser.add_argument(
        '--exp_path',
        help=f'path to log files folder (model checkpoints, tensorboard logs, \
            model outputs). (default: {os.path.join(LOGS_DIR, "training", "rf")})',
        default=os.path.join(LOGS_DIR, 'training', 'rf'))

    return parser


def nn_model_metrics_cli(description):
    """CLI for neural network model metrics."""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        'training_path',
        help='path to the training directory that holds yamls with history.',
        type=str)

    parser.add_argument(
        'report_path',
        help='path to report directory.',
        type=str,)

    return parser


def train_mars_mlr_cli(description):
    """CLI for Mars multiouput multiple linear regression."""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        'data_path',
        help='path to data (2D or 3D is fine).',
        type=str)

    parser.add_argument(
        '--exp_path',
        help=f'path to log files folder (model checkpoints, tensorboard logs, \
            model outputs).',
        type=str)

    return parser


def best_models_cli(description):
    """For retrieving information about the best models."""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        'yml_dir',
        help='path to directory containing YML files with `best_metric` attribute.',
        type=str)

    parser.add_argument(
        'out_dir',
        help='directory to write best hyperparameter information to. If the \
            supplied path does not exist, it will be created at the location if\
            possible.',
        type=str,)

    parser.add_argument(
        '--overwrite_out_dir',
        help='True to overwrite the contents of the out dir, False otherwise. \
            (default: False)',
        type=str,
        default='False')

    parser.add_argument(
        '--special_id',
        help='a unique identifier that would be used to select best \
            hyperparameters for specific models. (default: None)',
        type=str,
        default=None)

    return parser
