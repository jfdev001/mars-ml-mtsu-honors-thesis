"""Class for custom keras tuner that support cross validation.

https://kegui.medium.com/how-to-do-cross-validation-in-keras-tuner-db4b2dbe079a
https://github.com/keras-team/keras-tuner/issues/122
"""

import tensorflow as tf

import keras_tuner as kt
from keras_tuner.engine.tuner import Tuner

from sklearn.model_selection import KFold, TimeSeriesSplit

import numpy as np

import collections


class CVTuner(Tuner):
    """Tuner with cross validation splitters on data."""

    def __init__(self, log, cv='TimeSeriesSplit', n_splits=5,
                 iid=True, shuffle_batched_seq=False,
                 batch_size=32, epochs=1, drop_remainder=True, **kwargs):
        """Define state for CVTuner.

        :param log: Logger.
        :param cv: <class 'str'> that is `splitter.__class__.__name__` 
            of an child of the BaseCrossValidator sklearn class.
            Defaults to 'TimeSeriesSplit'.
        :param n_splits: <class 'int'> Parameter for sklearn 
            cross validation.
        :param iid_data: <class 'bool'> True if data is independent
            and identically distributed (iid), false otherwise. If the 
            data is not iid, then it is likely time series or another
            form of data that should be treated carefully.
        :param shuffle_sequential_batches: <class 'bool'> After sequential
            data has been batched, you could shuffle those batches. Only
            relevant if :param iid: is False.
        :param batch_size: <class 'int'>
        :param epochs: <class 'int'>
        :param drop_remainder: <class 'bool'>
        """

        # Inheritance
        super().__init__(**kwargs)

        # Validation
        if cv not in ['KFold', 'TimeSeriesSplit']:
            raise

        # Default sklearn splitter
        if cv == 'KFold':
            self.cv = KFold(n_splits=n_splits)

        elif cv == 'TimeSeriesSplit':
            self.cv = TimeSeriesSplit(n_splits=n_splits, test_size=batch_size)

        # Save args
        self.log = log
        self.n_splits = n_splits
        self.iid = iid
        self.shuffle_batched_seq = shuffle_batched_seq
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.epochs = epochs

    def run_trial(self, trial, x_train_val, y_train_val, **fit_kwargs):
        """Method for operations that occur during keras-tuner trials.

        NOTE: Unlike the base `Tuner` class, there is no model
        checkpointing for this subclass. This is because
        the best model (i.e., model weights) 
        will not be loaded using this class since the class's only 
        purpose is determine best hyperparameters for minimizing
        conditional Err_T (check elements of statistical learning for this)...

        :param trial: Required for subclassing `Tuner`.
        :param x_train_val: <class 'numpy.ndarray'> Feature dataset
            that contains both the training and validation data.
        :param y_train_val: <class 'numpy.ndarray'> Labels dataset
            that contains both the training and validation data.

        :return: <class 'NoneType'>
        """

        # Track metrics -- if a key does not exist, an empty list is
        # created by default
        metrics = collections.defaultdict(list)

        # Training and validation for each fold
        fold = 0
        for train_indices, validation_indices in self.cv.split(x_train_val):

            # Split indices for dataset...
            # NOTE: Batch size must be <= test size because validation
            # indices will be based on test_size arg for timeseries split
            x_train, x_val = x_train_val[train_indices], x_train_val[validation_indices]
            y_train, y_val = y_train_val[train_indices], y_train_val[validation_indices]

            # Make into tensorflow datasets
            train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            validation_data = tf.data.Dataset.from_tensor_slices(
                (x_val, y_val))

            # Determine shuffling strategy...
            # iid for TimeSeries means that within a training block, the
            # instances are independent and identically distributed
            # and the temporal dependence is maintained by virtue of
            # the block design itself
            if self.iid:

                # Shuffle data then batch
                train_data = train_data.shuffle(buffer_size=x_train.shape[0])
                train_data = train_data.batch(
                    batch_size=self.batch_size, drop_remainder=self.drop_remainder)

                validation_data = validation_data.shuffle(
                    buffer_size=x_val.shape[0])
                validation_data = validation_data.batch(
                    batch_size=self.batch_size, drop_remainder=self.drop_remainder)

            else:

                # Batch first then shuffle -- this is what TF RNN tutorial does?
                # TODO: Verify time_series from array does same as what I am doing
                # Maintains temporal dependence within windows but not between windows)

                # Batch non-iid data
                train_data = train_data.batch(
                    batch_size=self.batch_size, drop_remainder=self.drop_remainder)
                validation_data = validation_data.batch(
                    batch_size=self.batch_size, drop_remainder=self.drop_remainder)

                # Determine shuffling of the batched sequences
                if self.shuffle_batched_seq:
                    train_data = train_data.shuffle(
                        buffer_size=x_train.shape[0])

                    validation_data = validation_data.shuffle(
                        buffer_size=x_val.shape[0])

            self.log.info(f'fold: {fold}')
            self.log.info(f'{x_train.shape[0]} {x_val.shape[0]} \
                          {y_train.shape[0]} {y_val.shape[0]}')
            self.log.info(f'{train_data} {validation_data}')
            fold += 1

            # Model instantiation -- builds a new model for each train index
            model = self.hypermodel.build(trial.hyperparameters)

            # Model fitting
            history = model.fit(
                train_data,
                validation_data=validation_data,
                epochs=self.epochs,
                **fit_kwargs)

            # Tracking metrics -- 1 metric value per epoch
            for metric, epoch_values in history.history.items():

                # Determine min/max search objective
                if self.oracle.objective.direction == 'min':
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)

                # The metrics default dict will append the best value
                # for a given validation fold to a list whose key
                # is the metric returned by the fitting history
                metrics[metric].append(best_value)

        # Average results across cross validation splits
        averaged_metrics = {metric: np.mean(
            cv_fold_values) for metric, cv_fold_values in metrics.items()}

        # Update the oracle (the tuning algorithm) -- could update `step`
        # see `Tuner` engine for arg use
        self.oracle.update_trial(trial.trial_id, metrics=averaged_metrics, )

        # What is the utility of this step?
        self.save_model(trial.trial_id, model)
