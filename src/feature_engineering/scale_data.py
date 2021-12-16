"""Module for scaling data."""

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np


def scale_data(scaler, label_columns, train_df, val_df, test_df, return_joined_dfs=True):
    """Normalizes or standardizes data and returns the scaler.

    :param scaler: <class 'str'> 
    :param label_columns: <class 'list'>
    :param train_df: <class 'pandas.DataFrame'>
    :param val_df: <class 'pandas.DataFrame'>
    :param test_df: <class 'pandas.DataFrame'>
    :param return_joined_dfs: <class 'bool'> True to return train, val,
        and test dfs with combined features-labels, False otherwise.

    :return: <class 'tuple'>
        (1) scaled_train_df, 
        (2) scaled_val_df, 
        (3) scaled_test_df, 
        (4) scaled_train_features_df, 
        (5) scaled_val_features_df, 
        (6) scaled_test_features_df, 
        (7) scaled_train_labels_df, 
        (8) scaled_val_labels_df, 
        (9) scaled_test_labels_df, 
        (10) x_scaler, 
        (11) y_scaler
        OR
        (1) scaled_train_features_df, 
        (2) scaled_val_features_df, 
        (3) scaled_test_features_df, 
        (4) scaled_train_labels_df, 
        (5) scaled_val_labels_df, 
        (6) scaled_test_labels_df, 
        (7) x_scaler, 
        (8) y_scaler
    """

    # Append the train-validation sets to train the scaler
    train_val_df = train_df.append(val_df)

    # Returns a tuple where the first element an array in the
    # the combined dataframe where the indices are 0...
    # important because the training and validation dataframes
    # will have to be reseparated... the second element of the
    # array is where the validation data begins
    split_idxs = np.where(train_val_df.index.get_loc(0) == True)[0]

    # List of features
    feature_columns = [
        feature for feature in train_df.columns if feature not in label_columns]

    # Extract features
    train_val_features_df = train_val_df[feature_columns]
    test_features_df = test_df[feature_columns]

    # Extract labels
    train_val_labels_df = train_val_df[label_columns]
    test_labels_df = test_df[label_columns]

    # Determine scaler
    if scaler == 'std_scaler':
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
    elif scaler == 'minmax_scaler':
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
    else:
        raise ValueError(
            ':param scaler: must be `std_scaler` or `minmax_scaler`.')

    # Scaling
    x_scaler.fit(train_val_features_df)
    y_scaler.fit(train_val_labels_df)

    scaled_train_val_features_ndarr = x_scaler.transform(
        train_val_features_df)
    scaled_test_features_ndarr = x_scaler.transform(test_features_df)

    scaled_train_val_labels_ndarr = y_scaler.transform(train_val_labels_df)
    scaled_test_labels_ndarr = y_scaler.transform(test_labels_df)

    # # Save columns names -- since no feature/target extraction...
    # # these column names can be used for all dfs
    # columns = train_df.columns

    # Convert the feature ndarrays back to dataframes
    scaled_train_features_df = pd.DataFrame(
        scaled_train_val_features_ndarr[: split_idxs[1]])
    scaled_train_features_df.columns = feature_columns

    scaled_val_features_df = pd.DataFrame(
        scaled_train_val_features_ndarr[split_idxs[1]:])
    scaled_val_features_df.columns = feature_columns

    scaled_test_features_df = pd.DataFrame(scaled_test_features_ndarr)
    scaled_test_features_df.columns = feature_columns

    # Convert the label ndarrays back to dataframes
    scaled_train_labels_df = pd.DataFrame(
        scaled_train_val_labels_ndarr[: split_idxs[1]])
    scaled_train_labels_df.columns = label_columns

    scaled_val_labels_df = pd.DataFrame(
        scaled_train_val_labels_ndarr[split_idxs[1]:])
    scaled_val_labels_df.columns = label_columns

    scaled_test_labels_df = pd.DataFrame(scaled_test_labels_ndarr)
    scaled_test_labels_df.columns = label_columns

    # Join the features and label dataframes to create a composite
    # dataframe
    scaled_train_df = scaled_train_features_df.join(scaled_train_labels_df)
    scaled_val_df = scaled_val_features_df.join(scaled_val_labels_df)
    scaled_test_df = scaled_test_features_df.join(scaled_test_labels_df)

    # Return the scaled data and the scaler
    if return_joined_dfs:
        return scaled_train_df, scaled_val_df, scaled_test_df, \
            scaled_train_features_df, scaled_val_features_df, scaled_test_features_df, \
            scaled_train_labels_df, scaled_val_labels_df, scaled_test_labels_df, \
            x_scaler, y_scaler
    else:
        return scaled_train_features_df, scaled_val_features_df, scaled_test_features_df, \
            scaled_train_labels_df, scaled_val_labels_df, scaled_test_labels_df, \
            x_scaler, y_scaler
