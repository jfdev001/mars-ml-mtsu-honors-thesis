"""Module for keras-tuner hypermodels."""

import keras_tuner as kt
import tensorflow as tf

from . mars_nn import ConvMarsNN, SimpleMarsNN, MarsTransformer
from . mars_rf import build_mars_rf


class HyperModelBaseClass(kt.HyperModel):
    def __init__(self, model_obj, all_params):
        pass

    def build(self, hp):
        pass


class MultiOutputRFRHypermodel(kt.HyperModel):
    """HyperModel wrapper for random forest regressor."""

    def __init__(self, all_params):
        """Tuning lists or constants for random forest regressor.

        :param all_params: Nested dictionary with at least two parent keys --
            `model_params` and `compile_params`.
        """

        self.all_params = all_params

    def build(self, hp):
        """Build method that hypermodel uses during tuning"""

        # Cast list params in dict to hp choice
        params_to_hp(hp, all_params=self.all_params)

        # Build model
        model = build_mars_rf(**self.all_params['model_params'])

        # Return the result
        return model


class SimpleMarsNNHypermodel(kt.HyperModel):
    """HyperModel wrapper for SimpleMarsNN"""

    def __init__(self, all_params):
        """Tuning lists or constants.

        :param all_params: Nested dictionary with at least two parent keys --
            `model_params` and `compile_params`.
        """

        # Validate the all_params arg
        if not isinstance(all_params, dict):
            raise TypeError(':param all_params: should be <class `dict`>')

        elif isinstance(all_params, dict) \
                and ('model_params' not in all_params or 'fit_params' not in all_params):
            raise ValueError(
                ':param all_params: must have `model_params` and `fit_params` keys.')

        # Save args
        self.all_params = all_params

    def build(self, hp):
        """Build method that hypermodel uses during tuning."""

        # Cast list params in dict to hyperparameters
        params_to_hp(hp, self.all_params)

        # Instantiate model and compile model
        model = SimpleMarsNN(**self.all_params['model_params'])
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(**self.all_params['compile_params']))

        # Result of compilation and instantiation
        return model


class ConvMarsNNHypermodel(kt.HyperModel):
    """Hyper model wrpaper for ConvMarsNN."""

    def __init__(self, all_params):
        """Tuning lists or constants for ConvMarsNN.

        :param all_params: Nested dictionary with at least two parent keys --
            `model_params` and `compile_params`.
        """

        self.all_params = all_params

    def build(self, hp):

        # Cast list params in dict to hyperparameters
        params_to_hp(hp, self.all_params)

        # Instantitate and compile model
        model = ConvMarsNN(**self.all_params['model_params'])
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(**self.all_params['compile_params']))

        # Result of compilation and instantiation
        return model


class MarsTransformerHypermodel(kt.HyperModel):
    """"""

    def __init__(self, all_params):
        pass

    def build(self, hp):
        pass


def params_to_hp(hp, all_params):
    """Converts parameters in all params dict to hp choice if appropriate.

    :param hp: keras-tuner hyperparameter object.
    :param all_paramse: Nested dict.

    :return: None. Pass by obj-reference.
    """

    # Convert args that are lists greater than len 1 to
    # a hyperparameter choice
    for parent_key, parent_dict in all_params.items():
        for child_key, child_value in parent_dict.items():

            # Child must be list to be converted to hp choice
            if isinstance(child_value, list):

                # List len > 1 should be converted to hyperparameter choices
                if len(child_value) > 1:
                    parent_dict[child_key] = hp.Choice(
                        child_key, child_value)

                # List with single element is just a constant
                else:
                    parent_dict[child_key] = child_value[0]
