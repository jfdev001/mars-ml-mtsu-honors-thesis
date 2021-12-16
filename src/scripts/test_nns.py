"""Module for testing conv mars neural net.

For demonstration, use 

python test_conv_mars_nn.py ../../logs/training/conv_mars_nn_iid/20211018_14-57-13CST_history.yml \
    --task predict --which_dataset validation --exp_path ../../logs/testing/conv_mars_nn_iid \

in the CLI
"""


# System level


# Setup relative imports
import sys
import os
src = os.path.join(os.getcwd(), '..')
if src not in sys.path:
    sys.path.append(src)

# Relative imports
from cli import test_nns_cli  # nopep8
from models.nn_tester import NNTester  # nopep8
from models.mars_nn import ConvMarsNN  # nopep8
from models.mars_nn import SimpleMarsNN  # nopep8
from utils import cast_args_to_bool  # nopep8


if __name__ == '__main__':

    # CLI
    parser = test_nns_cli('testing neural nets')
    args = parser.parse_args()
    args = cast_args_to_bool(args)

    if args.history_path is not None:
        raise NotImplementedError('history path not in use...')

    # Select model
    if args.model == 'simple_mars_nn':
        model = SimpleMarsNN
    elif args.model == 'conv_mars_nn':
        model = ConvMarsNN

    # Trainer object
    nn_tester = NNTester(
        model=model,
        exp_path=args.exp_path,
        history_path=args.history_path,
        yml_path=args.yml_path)

    # Test neural net
    if args.task == 'cross_validate':
        nn_tester.walkforward_cv(
            which_dataset=args.which_dataset,
            n_splits=args.n_splits,
            rescale=args.rescale)
