"""Module for hyperparameter tuning conv mars nn"""


# System level
import sys
from pathlib import Path
import os

# TF and keras
from keras_tuner.tuners import BayesianOptimization
import tensorflow as tf
import keras_tuner as kt

# Time
import pytz
import datetime
import time

# Data wrangling
import numpy as np

# Misc
from copy import deepcopy
from collections import defaultdict

# Storage
import pickle
import yaml
from yaml.representer import Representer
yaml.add_representer(defaultdict, Representer.represent_dict)  # nopep8

# Setup relative imports
src = os.path.join(os.getcwd(), '..')
if src not in sys.path:
    sys.path.append(src)

# Relative imports
from models.nn_tuner import NNTuner  # nopep8
from models.mars_hypermodels import ConvMarsNNHypermodel  # nopep8
from cli import tune_conv_mars_nn_cli  # nopep8
import utils  # nopep8


if __name__ == '__main__':

    # CLI args
    parser = tune_conv_mars_nn_cli('Tune Simple Mars NN')

    # Instantiate the tuner wrapper class
    nn_tuner = NNTuner(parser=parser, hypermodel=ConvMarsNNHypermodel)

    # Tuning
    nn_tuner.tune()
