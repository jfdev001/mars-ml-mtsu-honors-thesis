"""Module for feature reduction either by backward elimination or pca."""

import os
import sys
from pathlib import Path

if not os.path.join(os.getcwd(), '..') in sys.path:
    sys.path.append(os.path.join(os.getcwd(), '..'))

import statsmodels as sm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import datetime
import pytz
import yaml
import pickle

from cli import DATA_DIR, LOGS_DIR
from utils import idx_of_substr_in_list  # nopep8


def backward_elimination(X_train, y_train, alpha):
    """Perform backward elimination to get only the best features.

    :param X_train: <class 'pandas.DataFrame'> The initial explanatory variable
        data that will be altered iteratively.
    :param y_train: <class 'pandas.DataFrame'> The response variable
        data that will remain constant throughout.
    :parma alpha: <class 'float'> The significance level.

    :return: A dataframe with only the most significant explanatory variables
        included.
    """

    # The initial data
    X_sm_train = sm.add_constant(X_train)

    # Do-While loop
    do = True
    while (do):
        # Model
        sm_model = sm.OLS(y_train, X_sm_train).fit()

        # Get the value and name of the feature with the highest p-value
        max_p_value = sm_model.pvalues.sort_values(ascending=False)[0]
        max_p_value_feature = sm_model.pvalues.sort_values(
            ascending=False).index.values[0]

        # Check if the feature should be removed
        if (max_p_value > alpha):
            X_sm_train = X_sm_train.drop(max_p_value_feature, axis=1)
        else:
            do = False

    # Return the optimized dataframe
    if ('const' in X_sm_train.columns):
        return X_sm_train.drop('const', axis=1)
    else:
        return X_sm_train


def reduce_features(args):
    """DEPRECATED

    :param :

    :return:
    """

    # Validate feature reduction parameter ranges
    perform_pca = False
    perform_backward_elimination = False
    if (args.pca_prop_explained_var >= 0) and (args.pca_prop_explained_var <= 1):
        perform_pca = True
    elif (args.backward_elimination_alpha >= 0) and (args.backward_elimination_alpha <= 1):
        perform_pca = True

    if not (perform_pca or perform_backward_elimination):
        raise ValueError(
            ':param backward_elimination_alpha: or :param pca_prop_explained_var: must be in [0.0, 1.0].')

    # Validate scaling
    if not 'tx' in args.data_path:
        raise ValueError(
            'Data must be formatted into feature and target split datasets.')
    elif 'no_scaler' in args.data_path:
        raise ValueError(
            'Data must be standardized. Make sure :param data_path: has `std_scaler`.')

    # Track datetime CST
    utc_begin = pytz.utc.localize(datetime.datetime.utcnow())
    cst_begin = utc_begin.astimezone(pytz.timezone("US/Central"))
    dtime = cst_begin.strftime('%Y%m%d_%H-%M-%S') + 'CST'

    # Load data
    print(f'\nLoad data {args.data_path}...')
    with open(args.data_path, 'rb') as fobj:
        data_dict = pickle.load(fobj)

    # Extract x and y data
    x_train, x_val, x_test, y_train, y_val, y_test, _ = data_dict.values()

    # Save where to split based on x_train size
    x_train_end_idx = x_train.shape[0]

    # Stack train and validation sets for x
    x_train_val = np.concatenate((x_train, x_val), axis=0)

    # Flatten the sets for PCA if the dataset is 3D
    if len(x_train_val.shape) == 3:
        x_train_val = x_train_val.reshape(x_train_val.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    # Determine feature reduction technique
    if args.pca_prop_explained_var != -1.0:
        # Fit on x_train + x_val then transform all sets
        print('\nFit PCA...')
        pca = PCA(n_components=args.pca_prop_explained_var)
        pca.fit(x_train_val)

        # Transform the sets
        print('\nPCA transform X...')
        pca_x_train_val_ndarr = pca.transform(x_train_val)
        pca_x_test_ndarr = pca.transform(x_test)

        # Restore train-val sets
        pca_x_train_ndarr = pca_x_train_val_ndarr[: x_train_end_idx]
        pca_x_val_ndarr = pca_x_train_val_ndarr[x_train_end_idx:]

        # Pickle datasets
        pickle_dict = dict(
            x_train=pca_x_train_ndarr,
            x_val=pca_x_val_ndarr,
            x_test=pca_x_test_ndarr,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            pca=pca)

        # Get name of file and split based on underscores
        fname = Path(args.data_path).name
        fname_split = fname.split('_')

        # The index-1 of the 'scaler' is the type of scaler used on the data
        scaler_idx = fname_split.index('scaler')
        scaler = f'{fname_split[scaler_idx-1]}_scaler'

        # Timesteps -- the last element of the substr is the tsteps
        x_timesteps = fname_split[idx_of_substr_in_list(fname_split, 'tx')]
        y_timesteps = fname_split[idx_of_substr_in_list(fname_split, 'ty')]

        # Check for negative sign
        if x_timesteps.find('-') == -1:
            x_timesteps = int(x_timesteps[-1])
        else:
            x_timesteps = -1 * int(x_timesteps[-1])

        if y_timesteps.find('-') == -1:
            y_timesteps = int(y_timesteps[-1])
        else:
            y_timesteps = -1 * int(y_timesteps[-1])

        # Modify args for logging purposes
        args.x_timesteps = x_timesteps
        args.y_timesteps = y_timesteps
        args.scaler = scaler
        args.overlapping_windows = 'True' if 'isoverlapping' in args.data_path else 'False'

        # The method
        reduce_method = 'pca'
        overlapping = 'isoverlapping' if 'isoverlapping' in args.data_path else 'nonoverlapping'

        # Save datasets
        save_path = os.path.join(
            DATA_DIR, f'{dtime}_{reduce_method}_{args.task}_{scaler}_{overlapping}_tx{x_timesteps}_ty{y_timesteps}_data_dict.pkl')
        with open(save_path, 'wb') as fobj:
            pickle.dump(pickle_dict, fobj)

    elif args.backward_elimination_alpha != -1.0:
        reduce_method = 'backward_elimination'
        raise NotImplementedError('BE not supported for multitarget data.')

    # Log data_cli args
    with open(os.path.join(args.exp_path, f'{dtime}_{reduce_method}_{args.task}_{overlapping}_args_log.yml'), 'w') as fobj:
        yaml.dump(args.__dict__, fobj)

    # Print log
    print(f'\n{args.task} complete.')
