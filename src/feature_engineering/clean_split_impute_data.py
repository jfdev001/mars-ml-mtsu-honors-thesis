"""Script for imputing data and splitting it to train, val, test splits."""

import os
import sys

if not os.path.join(os.getcwd(), '..') in sys.path:
    sys.path.append(os.path.join(os.getcwd(), '..'))

import datetime
import pytz

import pickle
import yaml

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

from cli import data_cli, DATA_DIR, LOGS_DIR
from sol_to_datetime import sol_to_datetime


def clean_split_impute_data(
        unclean_data_path,
        imputer,
        nan_threshold=1300,
        val_ratio=0.2, test_ratio=0.2,
        tree_random_state=42, iter_impute_random_state=42,
        n_jobs=-2, n_estimators=100, n_neighbors=5,
        max_impute_iter=10,
        skip_verification=False):
    """Cleans, train, val, splits; and imputes data.

    :param unclean_data_path: <class 'str'>
    :param imputer: <class 'str'>
    :param nan_threshold: <class 'int'>
    :param val_ratio: <class 'float'>
    :param test_ratio: <class 'float'>
    :param tree_random_state: <class 'int'>
    :param n_jobs: <class 'int'>
    :param n_estimators: <class 'int'>
    :param n_neighbors: <class 'int'>
    :param max_impute_iter: <class 'int'>
    :param skip_verification: <class 'bool'>

    :return: <class 'tuple'>
        (1) <class 'dict'> of <class 'pandas.DataFrame'>
        (2) <class 'str'> for impute estimator name
    """

    # Validate ratios
    if (val_ratio + test_ratio) >= 1.0:
        raise ValueError(
            'The sum of :param val_ratio: and :param test_ratio: must be less than or equal to 1.0')

    # Load base df
    print(f'\nLoading unclean data from {unclean_data_path}.')
    df = pd.read_csv(unclean_data_path)
    df = df.drop(labels=['Unnamed: 0'], axis=1)

    # Removal of large np.nan
    print('\nRemoving large np.nan and converting to timeseries.')
    nan_df = df.isna().sum()
    drop_labels = nan_df[nan_df > nan_threshold].index
    df = df.drop(labels=drop_labels, axis=1)

    # Get the sol dates in the df as datetime objs
    datetime_lst = sol_to_datetime(df['SOL'])

    # Set datetime and drop SOL
    df = df.set_index(pd.to_datetime(datetime_lst))
    df = df.drop(labels=['SOL'], axis=1)

    # Make the df contiguous time series (i.e., no gaps between days)
    df = df.asfreq('D')

    print('\nVariables in data...')
    print(df.columns)

    # Display ratios for user verification
    print('\nNumber of Samples:', df.shape[0])
    print('Ratios and samples -->')
    print('Test:', test_ratio,
          '--', int(np.ceil(test_ratio * df.shape[0])))
    print('Validation:', val_ratio,
          '--',  int(np.ceil(val_ratio * df.shape[0])))
    print('Train:', 1.0 - (val_ratio + test_ratio),
          '--', int(np.floor((1.0 - (val_ratio + test_ratio)) * df.shape[0])))

    # Verify ratios are okay
    if not skip_verification:
        proceed = input('Enter `y` or `Y` to continue: ')

        if not proceed.strip().lower() == 'y':
            print('Ratios are not satisfactory. Exiting script.')
            exit(0)

    # Initial split
    print('\nSplitting into train-validation and test sets.')
    df_train_val, df_test = train_test_split(
        df, test_size=test_ratio, shuffle=False)

    # Determine imputer
    # TODO:
    if imputer == 'bayesian_ridge':
        estimator = BayesianRidge()
    elif imputer == 'decision_tree':
        estimator = DecisionTreeRegressor(random_state=tree_random_state)
    elif imputer == 'extra_trees':
        estimator = ExtraTreesRegressor(
            n_jobs=n_jobs, n_estimators=n_estimators, random_state=tree_random_state)
    elif imputer == 'knn':
        estimator = KNeighborsRegressor(
            n_jobs=n_jobs, n_neighbors=n_neighbors)
    else:
        raise ValueError(
            ':param imputer: is not valid. Type `python <script_name> -h` for optional arguments.')

    # Perform multivariate imputation
    print(f'\nImputing with max impute iterations: {max_impute_iter}')
    iter_imputer = IterativeImputer(
        estimator=estimator, random_state=iter_impute_random_state, max_iter=max_impute_iter, verbose=True)
    iter_imputer.fit(df_train_val)

    # Validate continuation....

    # Tranform train-val and rename
    print('\nTransforming data using IterativeImputer')
    imputed_ndarray_train_val = iter_imputer.transform(df_train_val)
    imputed_df_train_val = pd.DataFrame()
    for ix, col_name in enumerate(df.columns):
        imputed_df_train_val[col_name] = imputed_ndarray_train_val[:, ix]

    # Transform test and rename
    imputed_ndarray_test = iter_imputer.transform(df_test)
    imputed_df_test = pd.DataFrame()
    for ix, col_name in enumerate(df.columns):
        imputed_df_test[col_name] = imputed_ndarray_test[:, ix]

    # Split the train val df into two separate dataframes
    print('\nSplit train-validation into train and validation sets.')
    imputed_df_train, imputed_df_val = train_test_split(
        imputed_df_train_val,
        test_size=(val_ratio/(1-test_ratio)),
        shuffle=False)

    # Reset index for val
    imputed_df_val = imputed_df_val.reset_index(drop=True)

    # Get estimator name
    est_name = estimator.__class__.__name__.lower()

    # Pickle it all up
    pickle_dict = {
        f'{est_name}_imputed_df_train': imputed_df_train,
        f'{est_name}_imputed_df_val': imputed_df_val,
        f'{est_name}_imputed_df_test': imputed_df_test, }

    # Display sizes of tvt sets for verification
    print('\nTrain Shape + %:', imputed_df_train.shape,
          imputed_df_train.shape[0]/df.shape[0])
    print('Validation Shape + %:', imputed_df_val.shape,
          imputed_df_val.shape[0]/df.shape[0])
    print('Test Shape + %:', imputed_df_test.shape,
          imputed_df_test.shape[0]/df.shape[0])

    if not skip_verification:
        # Get user input to check
        proceed = input('Enter `y` or `Y` to continue: ')

        if not proceed.strip().lower() == 'y':
            print('Process not satisfactory. Exiting script.')
            exit(0)

    # Return the pickled dictionary that is split and imputed
    return pickle_dict, est_name
