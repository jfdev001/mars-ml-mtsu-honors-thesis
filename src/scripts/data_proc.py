"""Script for handling data preprocessing.

Saving and logging should probably be controlled by script and
not within the called functions.
"""

import os
import sys
import pickle
from pathlib import Path

import pytz
import datetime

import numpy as np
from sklearn.decomposition import PCA

import yaml

if not os.path.join(os.getcwd(), '..') in sys.path:
    sys.path.append(os.path.join(os.getcwd(), '..'))

from feature_engineering.clean_split_impute_data import clean_split_impute_data
from feature_engineering.reduce_features import reduce_features
from feature_engineering.scale_data import scale_data
from feature_engineering.generate_timeseries_windows import generate_timeseries_windows

from distutils.util import strtobool

from cli import data_cli, DATA_DIR, LOGS_DIR  # nopep8
import utils  # nopep8

if __name__ == '__main__':

    # CLI args
    arg_parser = data_cli('Data Processing')
    args = arg_parser.parse_args()
    args = utils.cast_args_to_bool(args)
    args.labels = args.labels.split(',')

   # Track datetime CST
    utc_begin = pytz.utc.localize(datetime.datetime.utcnow())
    cst_begin = utc_begin.astimezone(pytz.timezone("US/Central"))
    dtime = cst_begin.strftime('%Y%m%d_%H-%M-%S') + 'CST'

    # Log default data path
    print(f'\n:param data_path: {args.data_path}')

    # Determine task
    if args.task == 'clean_split_impute':

        # Get the split and imputed pickle dictionary
        pickle_dict, est_name = clean_split_impute_data(
            unclean_data_path=args.unclean_data_path,
            imputer=args.imputer,
            nan_threshold=args.nan_threshold,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            tree_random_state=args.tree_random_state,
            n_jobs=args.n_jobs,
            n_estimators=args.n_estimators,
            n_neighbors=args.n_neighbors,
            max_impute_iter=args.max_impute_iter,
            skip_verification=args.skip_verification,)

        # Pickle data file name
        data_fname = f'{dtime}_{args.task}_tvt{(1.0 -(args.val_ratio+args.test_ratio))},{args.val_ratio},{args.test_ratio}_{est_name}_df_dict.pkl'

        # Log file name
        log_fname = f'{dtime}_{args.task}_tvt{(1.0 -(args.val_ratio+args.test_ratio))},{args.val_ratio},{args.test_ratio}_{est_name}_args_log.yml'

    # TODO: Refactor to function
    elif args.task == 'scale_rank3':

        # Load data that has features and labels predefined
        with open(args.data_path, 'rb') as fobj:
            train_df, val_df, test_df = pickle.load(fobj).values()

        # Determine scaling
        if args.scaler == 'no_scaler':
            scaler = None

        else:
            train_df, val_df, test_df, _, _, _, _, _, _, \
                x_scaler, y_scaler = scale_data(
                    scaler=args.scaler,
                    label_columns=args.labels,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df)

        # Generate the windows
        rank3_ndarrs_lst, rank2_dfs_lst = generate_timeseries_windows(
            feature_tsteps=args.x_timesteps,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_tsteps=args.y_timesteps,
            label_columns=args.labels,
            return_tstep_dfs=True,
            overlapping_windows=args.overlapping_windows)

        # Extract elements from lists
        x_train, x_train_df = rank3_ndarrs_lst[0], rank2_dfs_lst[0]
        x_val, x_val_df = rank3_ndarrs_lst[1], rank2_dfs_lst[1]
        x_test, x_test_df = rank3_ndarrs_lst[2], rank2_dfs_lst[2]
        y_train, y_train_df = rank3_ndarrs_lst[3], rank2_dfs_lst[3]
        y_val, y_val_df = rank3_ndarrs_lst[4], rank2_dfs_lst[4]
        y_test, y_test_df = rank3_ndarrs_lst[5], rank2_dfs_lst[5]

        # Log shapes
        print('\nWindow shapes:')
        print('Train:', x_train.shape, y_train.shape)
        print('Validation:', x_val.shape, y_val.shape)
        print('Test:', x_test.shape, y_test.shape)

        # Dictionary for the rank3 ndarrays and the scaler
        pickle_dict = dict(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            x_scaler=x_scaler,
            y_scaler=y_scaler)

        # Tracking overlapping vs. non-overlapping
        overlapping = 'isoverlapping' if args.overlapping_windows else 'nonoverlapping'

        # Pickle data file name
        data_fname = f'{dtime}_{args.task}_{args.scaler}_{overlapping}_tx{args.x_timesteps}_ty{args.y_timesteps}_data_dict.pkl'

        # Log file name
        log_fname = f'{dtime}_{args.task}_{overlapping}_tx{args.x_timesteps}_ty{args.y_timesteps}_args_log.yml'

    elif args.task == 'pca':
        # reduce_features(args)

        # Validate scaling
        if not 'tx' in args.data_path:
            raise ValueError(
                'Data must be formatted into feature and target split datasets.')

        elif 'std_scaler' not in args.data_path:
            raise ValueError(
                'Data must be standardized. Make sure :param data_path: has `std_scaler`.')

        # Load rank3 data
        with open(args.data_path, 'rb') as fobj:
            data_dict = pickle.load(fobj)

        # Extract x and y data
        x_train, x_val, x_test, y_train, y_val, y_test, _, _ = data_dict.values()

        # Save where to split based on x_train size
        x_train_end_idx = x_train.shape[0]

        # Stack train and validation sets for x
        print('\nConcatenating training and validation sets...')
        x_train_val = np.concatenate((x_train, x_val), axis=0)

        # Flatten the sets for PCA if the dataset is 3D
        if len(x_train_val.shape) == 3:
            print('\nReshaping rank-3 tensor to rank-2 tensor...')
            x_train_val = x_train_val.reshape(x_train_val.shape[0], -1)
            x_test = x_test.reshape(x_test.shape[0], -1)

        # Fit on x_train + x_val then transform all sets
        print('\nFit PCA...')
        pca = PCA(n_components=args.prop_explained_var)
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

        # Timesteps -- the third element on is the timesteps since the first
        # two characters of the string are tx or ty
        x_timesteps = fname_split[utils.idx_of_substr_in_list(
            fname_split, 'tx')][2:]
        y_timesteps = fname_split[utils.idx_of_substr_in_list(
            fname_split, 'ty')][2:]

        # Modify args for logging purposes
        args.x_timesteps = x_timesteps
        args.y_timesteps = y_timesteps
        args.scaler = scaler
        args.overlapping_windows = True if 'isoverlapping' in args.data_path else False

        # The method
        reduce_method = 'pca'
        overlapping = 'isoverlapping' if 'isoverlapping' in args.data_path else 'nonoverlapping'

        # Data file name
        data_fname = f'{dtime}_{args.task}_{scaler}_{overlapping}_tx{x_timesteps}_ty{y_timesteps}_data_dict.pkl'

        # Log file name
        log_fname = f'{dtime}_{args.task}_{scaler}_{overlapping}_tx{x_timesteps}_ty{y_timesteps}_args_log.yml'

    # Save and log info
    print('\nSaving and logging...')

    data_folder = Path(args.data_path).parent
    data_path = os.path.join(data_folder, data_fname)
    with open(data_path, 'wb') as fobj:
        pickle.dump(pickle_dict, fobj)

    log_path = os.path.join(args.exp_path, log_fname)
    with open(log_path, 'w') as fobj:
        yaml.dump(args.__dict__, fobj)

    print(f'\n{args.task} complete.')
