"""Module for time series window function."""

import pandas as pd
import numpy as np


def window_dataframe(df, steps, drop_t=False):
    """Shifts dataframe by desired steps with overlapping windows.

    :param df: <class 'pandas.DataFrame'>
    :param step: <class 'int'> Positive or negative. The number of
        previous or future timesteps to add as columns to the df.
    :param drop_t: <class 'bool'> True if dropping the column at
        timestep `_t`, i.e., the reference for t-1 or t+1 ... etc.
        Useful for visualizing how pandas shifting works.
    """

    # Copy
    df = df.copy()

    # Column names
    columns = df.columns

    # Calculate starting step, modifier for inclusivity, and the increment
    start_step = -1 if steps < 0 else 1

    # Drop timestep but modify inclusive range
    steps_modifier = 0
    if drop_t:
        steps_modifier = -1 if steps < 0 else 1

    step_increment = -1 if steps < 0 else 1

    # Iterate through label names
    for col_name in columns:

        # Shift labels by desired timestep
        for t in range(start_step, steps + steps_modifier, step_increment):

            # Determine the name of the new column
            col_name_at_shifted_tstep = col_name + "_" + \
                ("t+" if steps < 0 else "t-") + str(abs(t))

            # Update the dataframe in the list with a new column with the
            # shifted values
            df[col_name_at_shifted_tstep] = df[col_name].shift(t)

    # Rename any columns that didn't have timesteps
    df = df.rename(columns={col_name: (col_name + '_t')
                   for col_name in columns if col_name.find('_t') == -1})

    # Determine dropping timestep `_t` column.
    if drop_t:
        df = df.drop(columns=[col for col in df.columns if col.endswith('_t')])

    # Return the timestep shifted dataframe
    return df


def generate_timeseries_windows(
        feature_tsteps, label_tsteps, label_columns,
        train_df, val_df, test_df,
        return_tstep_dfs=True, overlapping_windows=True,):
    """Create windowed data in rank-2 or rank-3 format.

    Assume that feature tsteps is positive (i.e., previous days)
    while label tsteps is negative (i.e., future days)

    This function can make t-n or t+n sized windows where t+n is
    used for labeled data.  The user can specify returning the DF
    from which the rank-3 array is derived. Notably, the features
    are indices 0-2 of final lists while the labels are indices 3-5.
    The order within each subset of indices is train, validation, and
    testing data.

    The label dataframe will have timestep (i.e., label_t) dropped
    because then a feature dataframe will be predicting timesteps
    in the future... e.g., (x_t-2, x_t-1, x_t) -> (y_t+1, y_t+2)
    One might instead choose to formulate the problem as 
    e.g., (x_t-3, x_t-2, x_t-1) -> (y_t, y_t+1); HOWEVER, this function
    formulates the problem as the former and not the latter.

    :param feature_tsteps: <class 'int'> Previous (positive) timesteps 
        for features.
    :param label_tsteps: <class 'int'> Future (negative) timesteps 
        for labels.
    :param label_columns: <class 'list'>
    :param train_df: <class 'pandas.DataFrame'>
    :param val_df: <class 'pandas.DataFrame'>
    :param test_df: <class 'pandas.DataFrame'>
    :param return_tstep_dfs: <class 'bool'>
    :param drop_nontimestep_columns: <class 'bool'> Visualizing
        timestep shifts. Should set :param return_tstep_dfs: to True.
    :param overlapping_windows: <class 'bool'>

    :return: <class 'tuple'> if return_tstep_dfs else just numpy arrays.
        (1) <class 'list'> of length 6 of 
            <class 'numpy.ndarray'> elements representing
            x_train, x_val, x_test and y_train, y_val, y_test.
        (2) <class 'list'> of length 6 of 
            <class 'pandas.DataFrame'> elements of the train-val-test
            data.
    """

    # Validation
    if feature_tsteps < 1:
        raise ValueError(':param feature_tsteps: must be >= 1 (past shift).')
    if (label_tsteps is not None) and (label_tsteps >= 0):
        raise ValueError(':param label_tsteps: must be <= -1 (future shift).')
    if not isinstance(label_columns, list):
        raise TypeError(':param label_columns: must be <class `list`>')

    # List of features
    feature_columns = [
        feature for feature in train_df.columns if feature not in label_columns]

    # Make copies
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Extract features
    train_features_df = train_df[feature_columns]
    val_features_df = val_df[feature_columns]
    test_features_df = test_df[feature_columns]

    # Extract labels
    train_labels_df = train_df[label_columns]
    val_labels_df = val_df[label_columns]
    test_labels_df = test_df[label_columns]

    # Store dfs in list
    dfs = [train_features_df, val_features_df, test_features_df,
           train_labels_df, val_labels_df, test_labels_df]

    # Iterate through dfs and do time series shift
    for df_ix, df in enumerate(dfs):

        # Update the list of dataframes with the shifted dataframe
        dfs[df_ix] = window_dataframe(
            df,
            steps=feature_tsteps if df_ix in range(3) else label_tsteps,
            # `drop_t=True` for labels only
            drop_t=False if df_ix in range(3) else True,)

    # Store rank3 arrays
    rank3_ndarrs = [0 for i in range(len(dfs))]

    # Column names for timestep features and labels
    timestep_feature_columns = dfs[0].columns
    timestep_label_columns = dfs[-1].columns

    # list that will hold new features sets [: 3], and labels sets [3: ]
    cleaned_dfs = [0 for i in range(len(dfs))]

    # Clean the remaining dataframes and make the rank3 tensors
    for df_ix, df in enumerate(dfs):

        # Slice from the nan end of the features to the nan begin
        # of the labels... note subtracting one from feature tsteps
        # because drop_t does not occur for features... therefore
        # when feature_tsteps=3, then only 2 nan values arise rather
        # than 3... since label tsteps is a negative number, slicing
        # for negative number is still exclusive!
        # This also prevents data leakage since past t-step and
        # future t-steps that are out of range are filled with NAN,
        # rather than with the actual pre(pro)ceding information because
        # each df is separate and begins with 0 index.
        cleaned_dfs[df_ix] = df.iloc[
            feature_tsteps - 1: label_tsteps].reset_index(drop=True)

        # Column naming for the dataframes
        cleaned_dfs[df_ix].columns =  \
            timestep_feature_columns if df_ix in range(
                3) else timestep_label_columns

        # Determine overlapping df
        if not overlapping_windows:
            nonoverlap_df = pd.DataFrame()

            # `feature_tsteps` used as increment because it is the
            # driving factor behind overlap
            for i in range(0, len(cleaned_dfs[df_ix]), feature_tsteps):
                nonoverlap_df = nonoverlap_df.append(
                    cleaned_dfs[df_ix].iloc[i])

            # Update the dataframe list
            cleaned_dfs[df_ix] = nonoverlap_df.reset_index(drop=True)

        # Update the rank3 ndarray list
        tstep_dim = abs(feature_tsteps if df_ix in range(3) else label_tsteps)
        column_dim = len(feature_columns if df_ix in range(3)
                         else label_columns)
        rank3_ndarrs[df_ix] = cleaned_dfs[df_ix].to_numpy().reshape(
            -1, tstep_dim, column_dim)

    # Determine return of rank-3 tensors and/or the rank-2 dataframes
    if return_tstep_dfs:
        return rank3_ndarrs, cleaned_dfs
    else:
        return rank3_ndarrs
