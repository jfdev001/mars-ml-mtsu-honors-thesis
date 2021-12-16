"""Module for testing neural nets."""


# System level
from collections import defaultdict
import sys
import os
from pathlib import Path
from sklearn.utils import validation
import pandas as pd

# Data
import yaml

# Keras and tensorflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.keras.callbacks import EarlyStopping

# sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

# numpy
import numpy as np

# Setup relative imports
src = os.path.join(os.getcwd(), '..')
if src not in sys.path:
    sys.path.append(src)

# Relative imports
from models.nn_trainer import NNTrainer  # nopep8


class NNTester(NNTrainer):
    def __init__(
            self,
            model,
            exp_path,
            yml_path,
            history_path=None,
            allow_memory_growth=True):
        """Define state for NNTester.

        :param model: Keras model class (not instantiated obj).
        :param exp_path: <class 'str'> Logging path.
        :param history_path: <class 'str'> Path to yml file in training
            folder that holds model history... the model history
            results from a trained model with a particular set
            of hyperparameters... that yml hyperparameter file
            is nested in the training history and can be extracted.
        :param allow_memory_growth: <class 'bool'>
        """

        # Validate
        if history_path != None:
            raise NotImplementedError(
                'loading model weights via history is not in use...')

        # Save args
        self.exp_path = exp_path
        self.history_path = history_path

        # Logging
        self.log, self.dtime = super().setup_logging(
            exp_path=self.exp_path)

        # Get yaml and checkpoint path from history
        if history_path is not None:
            self.yml_path, self.ckpt_path = self.__get_paths_from_history(
                history_path=self.history_path)
        else:
            self.yml_path = yml_path

        # Get hyperparameters
        self.hparams = super().load_hparams(yml_path=self.yml_path)

        # Set special id if available
        self.special_id = self.hparams['optional arguments']['special_id']

        # Memory growth
        if allow_memory_growth:
            super().allow_memory_growth()

        # Use parent method to get data
        self.data = super().load_data(
            data_path=self.hparams['positional arguments']['data_path'])

        self.data.make_tf_datasets(
            iid=self.hparams['fit_params']['iid'],
            shuffle_batched_seq=self.hparams['fit_params']['shuffle_batched_seq'],
            batch_size=self.hparams['fit_params']['batch_size'],
            drop_remainder=self.hparams['fit_params']['drop_remainder'])

        # Set metrics
        self.metrics = super().set_metrics()

        # Load the model
        self.model = self.__load_model(model)

    def walkforward_cv(
            self,
            which_dataset: str,
            n_splits: int = None,
            rescale: bool = True):
        """Performs walk forward cross validation.

        Predicts on desired dataset (self.args.which_dataset),
        rescales the results, and then computes the desired
        evaluation statistics... (should have args for mse, rmse, mae, etc...

        Should not use the tf datasets for this.

        :param which_dataset: <class 'str'> Specifies whether the test
            set for CV is the holdout validation set ('test') or the
            validation set. If :param which_dataset: is 'test', then
            the training set is the train + validation set.
        :param n_splits: <class 'int'> Time Series Specific Arg.
        :param rescale: Whether to rescale the data or not.

        :return:
        """

        # Uses combined train-validation sets only for CV
        if which_dataset == 'validation':
            X = self.data.x_train_val
            Y = self.data.y_train_val

        # Uses combined train-validation-test set for CV...
        # this is for final model evaluation only
        elif which_dataset == 'test':
            X = np.concatenate(
                (self.data.x_train_val, self.data.x_test), axis=0)
            Y = np.concatenate(
                (self.data.y_train_val, self.data.y_test), axis=0)

        # Define callbacks for fitting -- only one needed is early
        # stopping...
        es = EarlyStopping(patience=self.hparams['fit_params']['es_patience'])

        # #
        # print('Testing model fit on arbitrary data')
        # x = np.random.normal(size=np.empty_like(X).shape)
        # y = np.random.normal(size=np.empty_like(Y).shape)
        # x_t = x[: x.shape[0]//2]
        # y_t = y[: x.shape[0]//2]
        # x_v = x[x.shape[0]//2:]
        # y_v = y[x.shape[0]//2:]
        # dummy_train = tf.data.Dataset.from_tensor_slices(
        #     (x_t, y_t)).shuffle(x_t.shape[0]).batch(32, drop_remainder=True)
        # dummy_val = tf.data.Dataset.from_tensor_slices((x_v, y_v)).shuffle(
        #     x_v.shape[0]).batch(32, drop_remainder=True)

        # print(x_t.shape[0])
        # print(x_v.shape[0])
        # print(dummy_train)
        # print(dummy_val)

        # self.model.fit(dummy_train, validation_data=dummy_val,
        #                epochs=32, verbose=0)
        # print('End training')
        # breakpoint()

        # Track scores for each fold
        scaled_scores = defaultdict(list)

        # Default n_splits
        if n_splits is None:
            n_splits = self.hparams['cv_params']['n_splits']

        # Set test size based on hyperparameters...
        # these two must be the same due to `drop_remainder` arg in
        # tf dataset preparation
        test_size = self.hparams['fit_params']['batch_size']

        # Validate n_splits
        max_n_splits = self.__max_n_splits_for_tseries(
            n_samples=X.shape[0], test_size=test_size)

        if n_splits > max_n_splits:
            raise ValueError(f':param n_splits: must be <= {max_n_splits}')

        # Time series cross validation
        cv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size)
        fold = 0
        es_stop_lst = []
        for train_indices, validation_indices in cv.split(X):

            # Split indices for dataset...
            x_train, x_val = X[train_indices], X[validation_indices]
            y_train, y_val = Y[train_indices], Y[validation_indices]

            # # TODO: Check dimensions
            # print(x_train.shape, x_val.shape)
            # breakpoint()

            # Determine if batching will be possible... this has to be done
            # because otherwise the dynamically built models like rnn
            # with seqs.shape[0] zeros tensor will fail.
            # if x_train.shape[0] >= self.hparams['fit_params']['batch_size'] \
            #         and x_val.shape[0] >= self.hparams['fit_params']['batch_size']:

            #     print('Pass:', x_train.shape[0], x_val.shape[0])
            #     breakpoint()

            # Make tf datasets
            train_samples = x_train.shape[0]
            fold_train = tf.data.Dataset.from_tensor_slices(
                (x_train, y_train))
            fold_train = fold_train.shuffle(train_samples).batch(
                batch_size=self.hparams['fit_params']['batch_size'],
                drop_remainder=True)

            test_samples = x_val.shape[0]
            fold_test = tf.data.Dataset.from_tensor_slices(
                (x_val, y_val))
            fold_test = fold_test.shuffle(test_samples).batch(
                batch_size=self.hparams['fit_params']['batch_size'],
                drop_remainder=True)

            # Logging dims
            self.log.info(f'fold: {fold}')
            self.log.info(f'{x_train.shape[0]} {x_val.shape[0]} \
                          {y_train.shape[0]} {y_val.shape[0]}')
            self.log.info(f'{fold_train} {fold_test}')
            fold += 1

            # print(fold_train)
            # print(fold_test)
            # breakpoint()

            # Fit the model
            fold_history = self.model.fit(
                fold_train, validation_data=fold_test,
                epochs=self.hparams['fit_params']['epochs'],
                callbacks=[es],)

            # Append early stopping to list for tracking purposes
            # https://stackoverflow.com/questions/50874596/how-to-detect-the-epoch-where-keras-earlystopping-occurred
            es_stop_lst.append(es.stopped_epoch)

            # Make predictions with model
            self.log.info(f'x_val: {x_val.shape}')
            preds = self.model(x_val)
            self.log.info(f'preds: {preds.shape}')

            # Rescaling
            if rescale:
                preds = self.__rescale(
                    data=preds, scaler=self.data.y_scaler)
                y_val = self.__rescale(
                    data=y_val, scaler=self.data.y_scaler)

            # Compute time related metrics
            for t in range(preds.shape[1]):
                for metric in self.metrics:
                    scaled_scores[f'{metric.__class__.__name__}_{t+1}'].append(
                        metric(y_val[:, t, :], preds[:, t, :]).numpy())

            # Flatten and check overall predictive capabilities
            for metric in self.metrics:
                scaled_scores[f'{metric.__class__.__name__}_overall'].append(metric(
                    y_val, preds).numpy())

                # # This is the same as the above calculation since tensorflow
                # # must automatically flatten the input along the last two dims
                # samples = y_val.shape[0]
                # scaled_scores[f'{metric.__class__.__name__}_flat'].append(metric(
                #     rescaled_targets.reshape(
                #         samples, -1), rescaled_preds.reshape(samples, -1)).numpy())
            # else:
            #     print('Fail:', x_train.shape[0], x_val.shape[0])

        # Compute summary statistics for metrics
        summary_scaled_scores = {}
        for metric_key, metric_fold_list in scaled_scores.items():
            summary_scaled_scores[f'{metric_key}_mean'] = np.mean(
                metric_fold_list)
            summary_scaled_scores[f'{metric_key}_std'] = np.std(
                metric_fold_list)

        # Write the raw data
        raw_data_fname = self.dtime + '_raw_fold_results.csv' if self.special_id is None else self.dtime + \
            f'_raw_fold_results-{self.special_id}.csv'
        raw_data_path = os.path.join(self.exp_path, raw_data_fname)
        self.__write_testing(path=raw_data_path, results=scaled_scores)

        # write summary data
        summary_data_fname = self.dtime + '_summary_fold_results.csv' if self.special_id is None else self.dtime + \
            f'_summary_fold_results-{self.special_id}.csv'
        summary_data_path = os.path.join(self.exp_path, summary_data_fname)
        self.__write_testing(path=summary_data_path,
                             results=summary_scaled_scores)

        # Write early stopping information
        es_scores = {
            'num_epochs': es_stop_lst,
            'avg': [np.mean(es_stop_lst)] + [np.nan for ele in range(len(es_stop_lst)-1)],
            'std': [np.std(es_stop_lst)] + [np.nan for ele in range(len(es_stop_lst)-1)]}
        es_fname = self.dtime + '_early_stopping_fold_results.csv' if self.special_id is None else self.dtime + \
            f'_early_stopping_fold_results.csv-{self.special_id}.csv'
        es_data_path = os.path.join(self.exp_path, es_fname)
        self.__write_testing(path=es_data_path, results=es_scores)

        # Task complete
        self.log.info(f'\nRescaled: {rescale}')
        self.log.info('\nEnd task!!')

    def __max_n_splits_for_tseries(
            self,
            n_samples: int,
            test_size: int,
            gap: int = 0) -> int:
        """Computes max n_splits for time series cross validation."""

        return (n_samples - gap)//test_size

    def __rescale(self, data: np.ndarray, scaler: MinMaxScaler):
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

    def __get_paths_from_history(self, history_path):
        """Extracts hyperparameter yml path and model checkpoint path."""

        # Get yml path to hyperparameters
        with open(history_path, 'r') as fobj:
            yml_path = yaml.safe_load(fobj)['yml_path']

        # Determine search criteria in training folder for
        # ckpts
        ckpt_path = None
        history_name = Path(history_path).name
        history_parent = Path(history_path).parent
        dtime_folder = history_name[: history_name.find(f'_history')]

        # Search for ckpts
        for item in os.listdir(history_parent):
            if item == dtime_folder:
                ckpt_path = os.path.join(
                    history_parent, item, 'ckpt')

        # Logging
        self.log.info('Path: ' + str(history_path))
        self.log.info('Name: ' + str(history_name))
        self.log.info('Parent: ' + str(history_parent))
        self.log.info('dtime_folder: ' + str(dtime_folder))
        self.log.info('ckpt_path ' + str(ckpt_path))

        return yml_path, ckpt_path

    def __load_model(self, model):
        """Prepares model based on checkpoint information and hparams.

        TODO: Allow loading of model weights directly instead of just
        compilation which is what is occurring currently.

        :param model: keras model class.

        :return: keras model object.
        """

        # Load model with no weights
        self.log.info('\nLoad model...')
        model = model(**self.hparams['model_params'])

        # Compile model
        # NOTE: Removed self.metrics from compilation
        self.log.info('\nCompile model...')
        model.compile(
            optimizer=Adam(
                learning_rate=self.hparams['compile_params']['learning_rate']),
            loss='mse',)

        # NOTE: Debugging removal of dummy call
        # # Dummy call to initialize random weights
        # self.log.info('\nDummy call for model...')
        # x_sample = self.data.x_train[: self.hparams['fit_params']
        #                              ['batch_size']]
        # result = model(x_sample)
        # self.log.info('Dummy output shape: ' + str(result.shape))

        # # Load model weights -- ckpt file path
        # self.log.info('\nLoading weights...')
        # self.log.info(f'Model type: {type(model)}')

        # # # NOTE: Can be used to inspect all bias/kernels are present...
        # # # except Adam optimizer info, which is NOT needed... therefore
        # # # use `expect_partial()` method
        # # print_tensors_in_checkpoint_file(
        # #     file_name=self.ckpt_path.__str__(), tensor_name='', all_tensors=False)

        # # From https://stackoverflow.com/questions/65057663/error-in-running-py-with-tensorflow-and-keras
        # # and https://github.com/digitalepidemiologylab/covid-twitter-bert/blob/master/run_predict.py
        # model.load_weights(self.ckpt_path).expect_partial()

        return model

    def __write_testing(self, path: str, results: dict) -> None:
        """Writes the testing result to csv.

        :param yml_dict: <class 'dict'> that will be written as
            a yml file.

        :return:
        """

        # Saving history as yaml
        self.log.info('\nSaving testing history...')

        # Raw lists
        if not isinstance(list(results.values())[0], list):
            df_results = pd.DataFrame(results, index=[0])
        else:
            df_results = pd.DataFrame(results)

        df_results.to_csv(path, index=False)

        return
