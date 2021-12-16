"""Class for tuning neural nets."""

# System level
import sys
from pathlib import Path
import os

# Time
import pytz
import datetime
import time

# Misc
from copy import deepcopy
import yaml

# TF and keras
import tensorflow as tf
import keras_tuner as kt

# Setup relative imports
src = os.path.join(os.getcwd(), '..')
if src not in sys.path:
    sys.path.append(src)

# Relative imports
from models.dataset import Dataset  # nopep8
from models.keras_cv_tuner import CVTuner  # nopep8
import utils  # nopep8


class NNTuner:
    """Neural Network Tuner interface"""

    def __init__(self, parser, hypermodel, allow_memory_growth=True):
        """Define state for NNTuner.

        :param args: CLI NameSpace
        :param model: Keras-tuner hypermodel object
        :param allow_memory_growth: <class 'bool'>
        """

        # Save init args
        self.parser = parser
        self.hypermodel = hypermodel

        # Extract cli arguments
        self.args, self.arg_groups_dict = self.__proc_args()

        # Initialize logging
        self.log, self.dtime = self.setup_logging(
            exp_path=self.args.exp_path)

        # Memory growth
        if allow_memory_growth:
            self.allow_memory_growth()

        # Get dataset obj
        self.data = self.load_data(data_path=self.args.data_path)

        # Tuner object
        self.tuner = None

        # Time elapsed in seconds
        self.time_dif = None

    def tune(self,):
        """Tune a desired hypermodel object.

        :return: <class 'NoneType'>
        """

        # Instantiate hyper model
        self.log.info('\nInstantiating hypermodel...')
        self.hypermodel = self.hypermodel(deepcopy(self.arg_groups_dict))

        # Early stopping callback
        es_callback = tf.keras.callbacks.EarlyStopping(
            patience=self.args.es_patience)

        # Time start
        start = time.time()

        # Determine search strat
        self.log.info(
            f'\nInstantiating tuner using {self.args.search_strategy} search strategy...')
        if self.args.search_strategy == 'cv_bopt':

            # Tuner instnatiation
            self.tuner = CVTuner(
                log=self.log,
                cv=self.args.cv,
                n_splits=self.args.n_splits,
                iid=self.args.iid,
                shuffle_batched_seq=self.args.shuffle_batched_seq,
                batch_size=self.args.batch_size,
                epochs=self.args.epochs,
                drop_remainder=self.args.drop_remainder,
                oracle=kt.oracles.BayesianOptimization(
                    objective='val_loss',
                    max_trials=self.args.max_trials),
                hypermodel=self.hypermodel,
                directory=self.args.exp_path,
                project_name=self.dtime,)

            # Log hps
            self.tuner.search_space_summary()

            self.log.info('\nHyperparameter search...')
            self.tuner.search(
                self.data.x_train_val, self.data.y_train_val,
                callbacks=[es_callback])

        elif self.args.search_strategy == 'bopt':

            # Make datasets
            self.data.make_tf_datasets(
                iid=self.args.iid,
                shuffle_batched_seq=self.args.shuffle_batched_seq,
                batch_size=self.args.batch_size)

            # Tuner instantiation
            self.tuner = kt.tuners.BayesianOptimization(
                hypermodel=self.hypermodel,
                objective='val_loss',
                max_trials=self.args.max_trials,
                executions_per_trial=self.args.executions_per_trial,
                directory=self.args.exp_path,
                project_name=self.dtime)

            # Log hps
            self.tuner.search_space_summary()

            # Search s
            self.log.info('\nHyperparameter search...')
            self.tuner.search(
                self.data.train_data,
                validation_data=self.data.validation_data,
                callbacks=[es_callback])

        # Time stop
        stop = time.time()

        # Difference in time
        self.time_dif = stop-start

        # Write experimental data to file
        self.__write_tuning()

    def __proc_args(self,):
        """Processes and extracts arguments from CLI.

        :return: <class 'tuple'> of <class 'argparser.NameSpace'> and 
            <class 'dict'>
        """

        # Parse CLI
        args = utils.cast_args_to_bool(self.parser.parse_args())

        # Extract arg groups
        arg_groups_dict = utils.get_arg_groups(args, self.parser)
        self.__set_output_tsteps_in_arg_groups_dict(args, arg_groups_dict)

        return args, arg_groups_dict

    def __set_output_tsteps_in_arg_groups_dict(self, args, arg_groups_dict):
        """Updates dict with output timesteps parameter from data path.

        :param args: <class 'argparser.NameSpace'>
        :param arg_groups_dict: <class 'dict'> of <class 'dict'> obj-ref.

        :return: <class 'NoneType'>
        """

        # Get tsteps for model
        data_path_as_lst = Path(args.data_path).name.split('_')
        output_tsteps_ix = utils.idx_of_substr_in_list(data_path_as_lst, 'ty')

        # TODO: Only works for output tsteps from 1 to 9, if multidigit
        output_tsteps = int(data_path_as_lst[output_tsteps_ix][-1])

        # Update args dict with the correct output timesteps --
        # this key is based on models/mars_nn.py
        arg_groups_dict['model_params']['output_tsteps'] = output_tsteps

        return None

    def setup_logging(self, exp_path):
        """Sets up logging and datetime.

        :return: <class 'tuple'> of log obj and dtime as string.
        """

        # Track datetime CST
        utc_begin = pytz.utc.localize(datetime.datetime.utcnow())
        cst_begin = utc_begin.astimezone(pytz.timezone("US/Central"))
        dtime = cst_begin.strftime('%Y%m%d_%H-%M-%S') + 'CST'

        # Instantiate logger obj
        log = utils.setup_logging(os.path.join(
            exp_path, f'{dtime}_info.log'))

        # Write to top of log
        log.info(dtime)

        return log, dtime

    def allow_memory_growth(self,):
        """Allows experimental gpu memory growth.

        :return: <class 'NoneType'>
        """

        # Memory allocation
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) != 0:
            self.log.info('\nUsing experimental memory growth.')
            tf.config.experimental.set_memory_growth(
                physical_devices[0], enable=True)
        else:
            self.log.info('\nNot using experimental memory growth.')

        return None

    def load_data(self, data_path):
        """Loads data for tuning.

        :return: <class 'Dataset'>
        """

        # Instantiate data obj
        self.log.info('\nLoading data...')
        data = Dataset(data_path=data_path)

        # Log data dimensions
        self.log.info(
            f'x_train: {data.x_train.shape} - y_train: {data.y_train.shape}')
        self.log.info(f'x_val: {data.x_val.shape} - y_val: {data.y_val.shape}')

        return data

    def __write_tuning(self,):
        """Writes results of experiment to file.

        :return: <class 'NoneType'>
        """

        # Mutate arg groups with tuner params
        self.__retrieve_params()

        # Dump the dictionary to yaml
        log_fname = f'{self.dtime}_{self.hypermodel.__class__.__name__}.yml'
        with open(os.path.join(self.args.exp_path, log_fname), 'w') as fobj:
            yaml.dump(self.arg_groups_dict, fobj)

        # Log completion
        self.log.info(
            f'\nTune {self.hypermodel.__class__.__name__}  complete.')

        return None

    def __retrieve_params(self):
        """Extracts hyperparameters from tuner and mutates arg_groups_dict.

        :return: <class 'NoneType'>
        """

        # Get hyperparameters
        self.log.info('\nRetrieving best hyperparameters...')
        tuner_best_hyperparameters = self.tuner.get_best_hyperparameters()[0].__dict__[
            'values']

        # Bes val loss
        min_trial_metrics = utils.get_best_trial_metric(
            os.path.join(self.args.exp_path, self.dtime), 'val_loss')

        # Abstract the best model process
        best_parameters = utils.get_best_parameters(
            arg_groups_dict=self.arg_groups_dict,
            tuner_best_hyperparameters=tuner_best_hyperparameters)

        # Update the groups dictionary
        self.arg_groups_dict['best_parameters'] = best_parameters
        self.arg_groups_dict['tuner_best_hyperparameters'] = tuner_best_hyperparameters

        self.arg_groups_dict['time_elapsed'] = f'{(self.time_dif)//(60*60)} hr \
        {(self.time_dif//60)} min \
        {int(float("." +str((self.time_dif)/60).split(".")[-1])*60)} s'

        self.arg_groups_dict['best_metric'] = min_trial_metrics

        return None
