"""Class for training neural nets"""

# System level
import sys
import os

# Data
import yaml

# Keras and tensorflow
import tensorflow as tf
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

# Setup relative imports
src = os.path.join(os.getcwd(), '..')
if src not in sys.path:
    sys.path.append(src)

# Relative imports
from models.nn_tuner import NNTuner  # nopep8


class NNTrainer(NNTuner):
    """Neural Network Trainer interface"""

    def __init__(self, parser, model, allow_memory_growth=True):
        """Define state for NNTrainer.

        :param parser: CLI NameSpace
        :param model: Keras model object
        :param allow_memory_growth: <class 'bool'>
        """

        # Save args
        self.parser = parser
        self.args = self.parser.parse_args()
        self.model = model

        # Get hyperparameter dictionary for model
        self.hparams = self.load_hparams(yml_path=self.args.yml_path)

        # Logging
        self.log, self.dtime = super().setup_logging(
            exp_path=self.args.exp_path)

        # Memory growth
        if allow_memory_growth:
            super().allow_memory_growth()

        # Use parent method to get data
        self.data = super().load_data(
            data_path=self.hparams['positional arguments']['data_path'])

        # Set a list of callbacks
        self.callbacks = self.__set_callbacks()

        # Set a list of metrics
        self.metrics = self.set_metrics()

        # Training history with special id and the yaml path
        self.history = None

    def train(self,):
        """Train-Validate model."""

        # Partition data
        self.data.make_tf_datasets(
            iid=self.hparams['fit_params']['iid'],
            shuffle_batched_seq=self.hparams['fit_params']['shuffle_batched_seq'],
            batch_size=self.hparams['fit_params']['batch_size'],
            drop_remainder=self.hparams['fit_params']['drop_remainder'])

        # Instantiate model
        self.log.info('\nInstantiating model...')
        self.model = self.model(**self.hparams['model_params'])

        # Compile model
        self.log.info('\nCompiling model...')
        self.model.compile(
            optimizer=Adam(
                learning_rate=self.hparams['compile_params']['learning_rate']),
            loss='mse',
            metrics=self.metrics)

        # Fit model
        self.log.info('\nFitting model...')
        history = self.model.fit(
            self.data.train_data,
            validation_data=self.data.validation_data,
            epochs=self.hparams['fit_params']['epochs'],
            callbacks=self.callbacks)

        # Add training history and unique identifier to a dict
        self.history = history.history
        self.history['special_id'] = \
            self.hparams['optional arguments']['special_id']
        self.history['yml_path'] = self.args.yml_path

        # Write data
        self.__write_training()

    def load_hparams(self, yml_path):
        """Loads hyperparameters for model from yaml."""

        with open(yml_path, 'r') as fobj:
            hparams = yaml.safe_load(fobj)['best_parameters']

        return hparams

    def __set_callbacks(self,):
        """TODO: Not modular."""

        early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=self.hparams['fit_params']['es_patience'])

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                self.args.exp_path,
                self.dtime, 'ckpt'),
            save_best_only=True, save_weights_only=True)

        tboard = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(
                self.args.exp_path,
                f'{self.dtime}_tboard'))

        callbacks = [early_stopping, checkpoint, tboard]

        return callbacks

    def set_metrics(self):
        """TODO: Not modular"""

        metrics = [MeanAbsoluteError(), MeanSquaredError(),
                   RootMeanSquaredError()]

        return metrics

    def __write_training(self,):
        """Writes the training history and special id to yaml."""

        # Saving history as yaml
        self.log.info('\nSaving training history...')

        with open(os.path.join(
                self.args.exp_path,
                f'{self.dtime}_history.yml'), 'w') as fobj:

            yaml.dump(self.history, fobj)

        # Task complete
        self.log.info('\nEnd task!!')
