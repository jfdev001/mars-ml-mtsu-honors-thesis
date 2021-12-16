"""Module for misc utils."""

import argparse
import datetime
import logging
import sys
import numpy as np
import scipy.stats as st
import os
import json
from distutils.util import strtobool
from copy import deepcopy
from collections import defaultdict
from matplotlib.axes import Axes
from matplotlib.container import BarContainer


def autolabel(
        ax: Axes,
        barcontainer: BarContainer,
        height_multiplier: float = 1.00,
        round_n: int = 2,
        x_mod: str = 'left',
        ha: str = 'left') -> None:
    """Labels the mean of the barcontainer above the container.

    https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
    """

    for ix, rect in enumerate(barcontainer):

        # Height for bar
        h = rect.get_height()

        # # Calculate x based on position of bar
        # if ix == 0:
        #     x = rect.get_x(),
        # elif ix == len(barcontainer):
        #     x = rect.get_x() + rect.get_width()/2
        # # else:
        # #     x = rect.get_x() + rect.get_width()

        # Set static x
        if x_mod == 'left':
            x = rect.get_x()
        elif x_mod == 'left_center':
            x = rect.get_x() + rect.get_width()/4
        elif x_mod == 'center':
            x = rect.get_x() + rect.get_width()/2
        elif x_mod == 'right_center':
            x = rect.get_x() + 3 * rect.get_width()/4
        elif x_mod == 'right':
            x = rect.get_x() + rect.get_width()
        elif isinstance(x_mod, float):
            x = rect.get_x() + x_mod
        else:
            raise ValueError(
                'invalid :param x_mod:. Must be in options above or float.')

        # Textbox
        ax.text(
            x=x,
            y=height_multiplier*h,
            s=str(round(h, round_n)),
            ha=ha,
            va='bottom')

    return None


def confidence_interval_err(vector: np.ndarray, alpha: float = 0.95):
    """Computes desired confidence interval error.

    TODO: Potential stack overflow question....

    According to this [excerpt](https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_confidence_intervals/bs704_confidence_intervals_print.html) from a Boston University biostatistics course, the left and right bound of a confidence interval can be calculated by
    ```
    right_bound = mean(sample) + 1.96 * standard_deviation(sample)/sqrt(len(sample))
    left_bound = mean(sample) - 1.96 * standard_deviation(sample)/sqrt(len(sample))
    ```

    While the explanation for this makes sense, implementing this by hand vs. using `scipy.stats` reveals some discrepancies in the outcome. **Why are these two intervals different??**

    ```python
    import numpy as np
    import scipy.stats as st

    data = np.random.randint(low=0, high=10, size=10)

    # Sample size < 30, therefore t-distro with deg. freedom len(data)-1
    # degree of freedom justification: https://en.wikipedia.org/wiki/Confidence_interval
    # scale = std error of mean justification: https://stats.stackexchange.com/questions/197676/why-do-t-test-use-standard-error-and-not-standard-deviation
    scipy_interval = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem())

    # Calculating 
    ```
    """

    # Validate
    if not(alpha > 0 and alpha < 100):
        raise ValueError(':param alpha: must be between 0 and 100.')

    # Compute estimators
    mean = np.mean(vector)
    scale = st.sem(vector)  # standard error mean (how close to pop. mean)

    # Determine central limit theorem assumption
    if vector.shape[0] >= 30:
        print('Assume CLT...')
        interval = st.norm.interval(alpha=alpha, loc=mean, scale=scale)

    else:
        print('Not assuming CLT...')
        interval = st.t.interval(alpha=alpha, df=len(
            vector)-1, loc=mean, scale=scale)

    # Confidence intervals are m +- h
    # from left_bound = mean - h and right_bound = mean + h
    # From the 'Example' at https://en.wikipedia.org/wiki/Confidence_interval
    # it is clear that [mean - cs/sqrt(n), mean + cs/sqrt(n)] means that the
    # value of cs/sqrt(n), denoted err can be computed with simple rearrangement.
    left_bound, right_bound = interval
    err = right_bound - mean

    # Resulting err for errorbars
    return err


def get_best_parameters(arg_groups_dict, tuner_best_hyperparameters):
    """Merges the nested dictionary parameters with the tuner's.

    :param arg_groups_dict: <class 'dict'> of <class 'dict'> that
        holds the CLI argument groups and their values.
    :param tuner_best_hyperparameters: <class 'dict'> The result
        of the keras-tuner hyperparameter search process.

    :return: <class 'dict'>
    """

    # For storing the constants only (i.e., no list dictionary)
    best_parameters = defaultdict(dict)

    # Iterate through arg groups and if the value is a dictionary...
    # check to see if any keys for that value are in best hyperparameters
    # see each value corresponds to a group key (e.g., model_params = {...}),
    # then the group key can be used to update the best parameters
    # dictionary
    for i_key, i_val in arg_groups_dict.items():
        if isinstance(i_val, dict):
            for j_key, j_val in i_val.items():

                # Parameter in best hyperparameters
                if j_key in tuner_best_hyperparameters:
                    best_parameters[i_key][j_key] = tuner_best_hyperparameters[j_key]

                # Parameter in arg groups and list of length 1
                if isinstance(j_val, list) and len(j_val) == 1:
                    best_parameters[i_key][j_key] = j_val[0]

                # Parameter in arg groups and just a constant (e.g. output_tsteps)
                elif not isinstance(j_val, list):
                    best_parameters[i_key][j_key] = j_val

    # The reorganized dictionary
    return best_parameters


def cast_args_to_bool(args: argparse.Namespace) -> dict:
    """Converts True/False strings in argument dictionary to bool.

    # TODO: Could replace any boolean args with lambda type
    # thus removing the need for this call.

    :param: command line object.

    :return: The modified args.
    """

    copied_args = deepcopy(args)

    for dest, value in copied_args.__dict__.items():
        if isinstance(value, list) \
                and ('True' in value or 'False' in value):
            copied_args.__dict__[dest] = [
                bool(strtobool(val)) for val in value]

        elif value == 'True' or value == 'False':
            copied_args.__dict__[dest] = bool(strtobool(value))

    return copied_args


def get_best_trial_metric(path, metric):
    """For a given tuning project, extracts the best val loss.

    This val loss corresponds to the best hyperparameters.

    :param path: <class 'str'>

    :return: <class 'float'>
    """

    # Extract best metric
    trial_metrics = []

    # Get directories
    project_dir = path
    trial_dirs = [f for f in os.listdir(project_dir) if f.find('trial') != -1]

    # Iterate through dirs that have trial.json in them
    for trial_dir in trial_dirs:

        # Open json
        with open(os.path.join(project_dir, trial_dir, 'trial.json'), 'rb') as fobj:

            # Extract val loss from json
            best_metric = json.load(
                fobj)['metrics']['metrics'][metric]['observations'][0]['value'][0]

            # Append to list
            trial_metrics.append(best_metric)

    # Min of list
    min_trial_metrics = min(trial_metrics)

    # The minimum of all trials for a particular project
    return min_trial_metrics


def get_arg_groups(args, parser):
    """Returns a dictionary of argument groups from CLI object.

    See https://github.com/python/cpython/blob/3.9/Lib/argparse.py

    :param args: <class 'argparse.Namespace'> 
    :param parser: <class 'argparse.ArgumentParser'>

    :return: <class 'dict'>
    """

    arg_groups = {}
    for group in parser._action_groups:

        # Each _ArgumentGroup (positional, optional, all other groups..)
        # object has a number of actions associated
        # with it and each Action has a 'dest' attribute that holds
        # the name of the attribute
        # In other words: The key 'a.dest' is the name of the group
        # itself. Then the group itself has group actions (i.e.,
        # args associated with the group)
        # dest is the name such as `epochs` in `--epochs`
        # a.dest -> str for the action destiation (e.g., 'epochs')
        # while args is the Namespace object
        group_dict = {a.dest: getattr(args, a.dest, None)
                      for a in group._group_actions}

        # A given group has a title in addition to its actions
        arg_groups[group.title] = group_dict

    return arg_groups


def setup_logging(log_path):
    """Logging information.

    :param log_path: <class 'str'>
    :param dtime: <class 'str'> Datetime 

    :return: Log object.
    """

    log = logging.getLogger()

    log.setLevel(logging.INFO)

    file_out = logging.FileHandler(log_path)

    stdout = logging.StreamHandler(sys.stdout)

    log.addHandler(file_out)
    log.addHandler(stdout)

    return log


def idx_of_substr_in_list(lst, substr):
    """Gets the index of the desired substring in a list of strings."""

    not_found = True
    cnt = 0
    ix = -1
    while (not_found) and (cnt < len(lst)):
        if substr in lst[cnt]:
            not_found = False
            ix = lst.index(lst[cnt])
        cnt += 1

    return ix


# def sol_to_datetime(sol_series):
#     """Convert SOL to datetime format.

#     Steps in the algorithm:
#     (1) Get previous and current sol in sol_series.
#     (2) Find the difference between the two.
#     (3) Append the sum of the previous datetime
#         element and a timedelta where delta
#         is determined by the difference from
#         step 2.
#     Note: This is technically an approximation
#     since SOLs are slightly longer than Earth days.

#     :param sol_series: <class 'pandas.core.series.Series'> containing integers
#         representing SOL days. For example, at index 0 in the series, the
#         value is 1 (meaning SOL 1). Since there are skips in the data,
#         the difference between the previous and current SOL value can be
#         converted to a time delta and added to the rover data which is
#         set at the beginning of the algorithm.
#     :return:  <class 'pandas.core.series.Series'> The corrected series in
#         <class 'datetime.date'> format
#     """
#     # Date of initial collection from https://mars.nasa.gov/msl/mission/overview/
#     rover_start_date = datetime.date(year=2012, month=8, day=5)

#     # Build new list with for loop
#     datetime_list = [rover_start_date]
#     for ix in range(1, len(sol_series)):

#         # Previous SOLs in list
#         prev_sol = sol_series.values[ix - 1]

#         # Current SOLs in list
#         cur_sol = sol_series.values[ix]

#         # The difference between the two SOLs
#         diff = cur_sol - prev_sol

#         # Appending to the datetime_list
#         datetime_list.append(
#             datetime_list[ix - 1] + datetime.timedelta(days=int(diff)))

#     # Concatenate list to the series and return
#     return datetime_list
