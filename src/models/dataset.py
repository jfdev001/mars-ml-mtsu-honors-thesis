"""Class dataset operations."""

import pickle
import tensorflow as tf
import numpy as np


class Dataset:
    """Dataset interface.

    TODO: Make property decorators with return value type
    so that suggestions for numpy or tensorflow datatypes
    are available.
    """

    def __init__(self, data_path: str):
        """Define state for Dataset.

        :param data_path: Path to data dictionary of x-y train-val-test pkl
        """

        # Base ndarrays and scaler -- based on script/data_proc.py
        # 'scale_rank3'
        with open(data_path, 'rb') as fobj:
            self.x_train, self.x_val, self.x_test, \
                self.y_train, self.y_val, self.y_test, \
                self.x_scaler, self.y_scaler = pickle.load(fobj).values()

        # Concatenated ndarrays
        self.x_train_val = np.concatenate((self.x_train, self.x_val), axis=0)
        self.y_train_val = np.concatenate((self.y_train, self.y_val), axis=0)

        # Tensorflow datasets
        self.train_data = None
        self.validation_data = None
        self.test_data = None

    def flatten_3D(self,) -> None:
        """Converts 3D data to 2D data."""

        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_val = self.x_val.reshape(self.x_val.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)

        self.y_train = self.y_train.reshape(self.y_train.shape[0], -1)
        self.y_val = self.y_val.reshape(self.y_val.shape[0], -1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0], -1)

        self.x_train_val = self.x_train_val.reshape(
            self.x_train_val.shape[0], -1)
        self.y_train_val = self.y_train_val.reshape(
            self.y_train_val.shape[0], -1)

        return None

    def rescale(self, ) -> None:
        """Rescales 3D data for use with linearity tests."""

        if len(self.x_train.shape) != 3:
            raise ValueError('data must be 3D to be rescaled.')

        # Use x timesteps dimension (i.e., (samples, timesteps, features))
        for t in range(self.x_train.shape[1]):
            self.x_train[:, t, :] = self.x_scaler.inverse_transform(
                self.x_train[:, t, :])

            self.x_val[:, t, :] = self.x_scaler.inverse_transform(
                self.x_val[:, t, :])

            self.x_test[:, t, :] = self.x_scaler.inverse_transform(
                self.x_test[:, t, :])

        # Use y timesteps dimension
        for t in range(self.y_train.shape[1]):
            self.y_train[:, t, :] = self.y_scaler.inverse_transform(
                self.y_train[:, t, :])

            self.y_val[:, t, :] = self.y_scaler.inverse_transform(
                self.y_val[:, t, :])

            self.y_test[:, t, :] = self.y_scaler.inverse_transform(
                self.y_test[:, t, :])

        # Reconcatenate the train_val sets with the scaled data
        self.x_train_val = np.concatenate((self.x_train, self.x_val), axis=0)
        self.y_train_val = np.concatenate((self.y_train, self.y_val), axis=0)

        return None

    def make_tf_datasets(
            self,
            iid=False,
            shuffle_batched_seq=False,
            train_buffer=None, validation_buffer=None, test_buffer=None,
            batch_size=32, drop_remainder=True):
        """Mutates self and creates training, validation, and testing tf datasets.

        :param iid: <class 'bool'> Independent, identically distributed
            data.
        :param shuffle_batched_seq: <class 'bool'> Only relevant if
            `:param iid:` is False.
        :param train_buffer: <class 'int'>
        :param validation_buffer: <class 'int'>
        :param test_buffer: <class 'int'>
        :param batch_size: <class 'int'>
        :param drop_remainder: <class 'bool'>

        :return: <class 'NoneType'>
        """

        # Set buffer
        if train_buffer is None:
            train_buffer = self.x_train.shape[0]
        if validation_buffer is None:
            validation_buffer = self.x_val.shape[0]
        if test_buffer is None:
            test_buffer = self.x_test.shape[0]

        # Init datasets
        self.train_data = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.y_train))

        self.validation_data = tf.data.Dataset.from_tensor_slices(
            (self.x_val, self.y_val))

        # TODO: To plot actual vs preds, cannot use the tf data,
        # should just use the base ndarrays that are available in
        # __init__(self)
        self.test_data = tf.data.Dataset.from_tensor_slices(
            (self.x_test, self.y_test))

        # Determine shuffling and dataset partitioning strategies
        if iid:

            # Shuffle instances and then batch the shuffled instances
            self.train_data = self.train_data.shuffle(
                train_buffer).batch(batch_size, drop_remainder)

            self.validation_data = self.validation_data.shuffle(
                validation_buffer).batch(batch_size, drop_remainder)

            # NOTE: The model is not fitted here but the random
            # sampling will allow for more robust metrics
            self.test_data = self.test_data.shuffle(
                test_buffer).batch(batch_size, drop_remainder)

        else:

            # Batch data
            self.train_data = self.train_data.batch(
                batch_size, drop_remainder)

            self.validation_data = self.validation_data.batch(
                batch_size, drop_remainder)

            self.test_data = self.test_data.batch(batch_size, drop_remainder)

            # Determine shuffling of batched sequential data
            if shuffle_batched_seq:
                self.train_data = self.train_data.shuffle(train_buffer)

                self.validation_data = self.validation_data.shuffle(
                    validation_buffer)

                self.test_data = self.test_data.shuffle(test_buffer)

        return None
