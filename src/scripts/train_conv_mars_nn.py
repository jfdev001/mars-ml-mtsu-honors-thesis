"""Module for training conv mars nn"""

# System level
import sys
import os


# Setup relative imports
src = os.path.join(os.getcwd(), '..')
if src not in sys.path:
    sys.path.append(src)

# Relative imports
from cli import train_nns_cli  # nopep8
from models.nn_trainer import NNTrainer  # nopep8
from models.mars_nn import ConvMarsNN  # nopep8


if __name__ == '__main__':

    # CLI
    parser = train_nns_cli('training convolutional mars neural net')

    # Trainer object
    nn_trainer = NNTrainer(parser=parser, model=ConvMarsNN, )

    # Train neural net
    nn_trainer.train()
