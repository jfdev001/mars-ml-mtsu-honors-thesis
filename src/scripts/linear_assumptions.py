"""Script to computing inferential statistics for lin. reg. assumptions.

Will require both features and targets since OLS may need to be
performe to acquire regression statistics. OLS for each timestep
in the forecast horizon will also need to be computed such that a
single OLS model is built for each of the 7-days in the future.

How could inter-residual Durbin-Watson be computed to show serial
correlation between errors??? If at all

Check Verbeek Ch.2 for Gauss-Markov assumptions.

The results of these statistical tests could be compiled into a table
such as

    All VIF > 10    | Hypothesis Test Results for Heteroskedasticity
    -------------------------------------------
t+1                 |
t+2
t+3
...
t+7

Possible tests auto-correlation durbin_watson,
breusch-pagan test or White Test for heterosked, nonlinearity linear_harvey_collier,
multicollinearity VIF.

Note the term 'exogenous' is a term in econometrics that refers to independent
variables (or features).

# Statsmodels on Regression Diagnostics and Specification (Assumption)  Tests
https://www.statsmodels.org/stable/diagnostic.html
https://www.statsmodels.org/dev/examples/notebooks/generated/regression_diagnostics.html

# Stack Exchange Questions
https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
https://stats.stackexchange.com/questions/496034/whats-the-difference-between-multicollinearity-and-autocorrelation

```python
python linear_assumptions.py \
    ../../data/20211103_11-31-57CST_scale_rank3_minmax_scaler_isoverlapping_tx28_ty-7_data_dict.pkl
```
"""

from __future__ import annotations

import argparse
from enum import auto
import os
import pickle
import sys

if not os.path.join(os.getcwd(), '..') in sys.path:
    sys.path.append(os.path.join(os.getcwd(), '..'))

import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_goldfeldquandt, spec_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.linear_model import RegressionResultsWrapper


from tqdm import tqdm

from models.dataset import Dataset  # nopep8


def fit_models(
    y_timesteps: list[np.ndarray],
    x: np.ndarray) -> tuple[
        list[OLS],
        list[RegressionResultsWrapper],
        list[np.ndarray],
        np.ndarray]:
    """Fit models given x (with constant) and y data."""

    # Instantiate OLS objects for each step in forecast horizon
    models = []
    reg_results = []
    resids = []
    for y_t in tqdm(y_timesteps, desc='Fitting models'):

        # Model instantiation
        model = sm.OLS(y_t, x)

        # Fit OLS model
        reg_result = model.fit()

        # Residuals
        resid = reg_result.resid

        # Append objs to lists
        models.append(model)
        reg_results.append(reg_result)
        resids.append(resid)

    # Exogenous variables -- same for all models
    exog = reg_result.model.exog

    # Return vars
    return models, reg_results, resids, exog


def compute_multicolinearity(
        exog: np.ndarray,
        vif_threshold: int) -> bool or str:
    """Compute VIFs and VIFs > threshold means multicolinearity exists."""

    # VIF -- multicolinearity
    # Exogenous variables are all the same...??
    vifs = np.array([variance_inflation_factor(
        exog, exog_idx=ix) for ix in tqdm(range(exog.shape[1]), desc='Computing VIFs')])

    # If all the variables VIFs for each feature are greater than
    # the threshold, then multicolinearity exists
    multicolinearity = np.all(vifs > vif_threshold)

    if not multicolinearity:

        # Count the number of cases for which vifs were greater than
        # the threshold and therefore the number of explanatory variables
        # that are colinear with at least one other explanatory variable
        multicolinearity = f'{np.count_nonzero(vifs > vif_threshold)}/{exog.shape[1]}'

    return multicolinearity


def compute_homoskedasticity(
        y_timesteps: list[np.ndarray],
        x: np.ndarray,
        alpha: float) -> bool:
    """Uses Goldfeld-Quandt test to reject H0 that variance is heteroskedastic."""

    # Goldfeld=Quandt -- Heteroskedasticity
    # H0=Variance in one subsample is larger than the other
    # therefore failing to reject H0 implies Heteroskedastic variance
    gq_stats_all_targets = []
    for y_t in tqdm(y_timesteps, desc='Goldfeld-Quandt Tests'):

        # Compute stats -- note gq stats are all strings
        gq_stats = het_goldfeldquandt(y=y_t, x=x)

        # Append to list for all steps in forecast horizon
        gq_stats_all_targets.append(gq_stats)

    # Numpy array it
    gq_stats_all_targets = np.array(gq_stats_all_targets)

    # # Make dataframe
    # gq_stats_df = pd.DataFrame({
    #     'fval': gq_stats_all_targets[:, 0],
    #     'pval': gq_stats_all_targets[:, 1], }, dtype=np.float64)

    # Extract p-values
    gq_pvals = gq_stats_all_targets[:, 1].astype(np.float64)

    # Determine homoskedasticity
    homoskedastic = np.all(gq_pvals < alpha)

    return homoskedastic


def compute_autocorrelation(resids: list[np.ndarray]) -> bool:
    """Uses Durbin-Watson test to reject H0 that there is no residual autocorrelation."""

    raise NotImplementedError(
        'cannot justify design here without further investigation.')

    # Concatenate the residuals
    concatenated_resids = np.array(resids).flatten()

    # Reshape to row major
    all_resids = np.array(resids)
    all_resids = all_resids.reshape(-1, all_resids.shape[0])  # N x M

    print(concatenated_resids.shape)
    print(all_resids.shape)

    # # Durbin-watson -- autocorrelation
    # # one result for each response variable
    # dw = np.array([durbin_watson(resids=resid)
    #               for resid in tqdm(resids, desc='Durbin-Watson Tests')])

    # Play around with axes?? axis=1??? for (n, timesteps)
    dw_concat = durbin_watson(resids=concatenated_resids)
    dw_time_major = durbin_watson(resids=all_resids, axis=1)

    print(dw_concat)
    print(dw_time_major)
    breakpoint()

    # print(all_resids.shape)
    # dw = np.array(durbin_watson(resids=all_resids))

    # # If all values in dw < alpha, reject H0 that no serial (auto) correlation
    # # between residuals
    # print(dw)

    # # Average distance of each dw stat to 2 since 2 indicates that there
    # # is no serial correlation
    # # https://www.statsmodels.org/dev/generated/statsmodels.stats.stattools.durbin_watson.html
    # autocorrelation = np.linalg.norm(
    #     dw - np.full(shape=dw.shape, fill_value=2))

    #  np.full(shape=dw.shape, fill_value=2)

    # print(autocorrelation)
    # breakpoint()

    return autocorrelation


def compute_linearity(
        resids: list[np.ndarray],
        exog: np.ndarray,
        alpha: float) -> bool:
    """Uses White's two-moment specification test for homosked. and linearity."""

    raise NotImplementedError('intractable for high dimensional data..')

    spec_white_stats = []
    for resid in tqdm(resids, desc='White`s Two-Moment Specification Tests'):
        spec_white_stat = spec_white(resid=resid, exog=exog)
        spec_white_stats.append(spec_white_stat)
    spec_white_stats = np.array(spec_white_stats)

    spec_white_pvals = spec_white_stats[:, 1]

    print(spec_white_pvals)
    breakpoint()

    homoskedastic_and_linear = np.all(spec_white_pvals < alpha)

    return homoskedastic_and_linear


if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser(
        'compute inferential statistics using training (train+val) data.')

    parser.add_argument(
        'read_path',
        help='cleaned, split, and imputed data path',
        type=str)

    # parser.add_argument(
    #     '--task',
    #     help='test to conduct for lin. reg assumption. \
    #         Multicolinearity uses VIF, autocorrelation uses Durbin-Watson\
    #         heterosked(asticity) uses Goldfeld-Quandt, and \
    #             nonlinearity uses Harvey Collier. \
    #             (default: None -> performs all tests)',
    #     choices=[
    #         'multicolinearity',
    #         'autocorrelation',
    #         'heterosked',
    #         'nonlinearity', ],
    #     default=None)

    parser.add_argument(
        '--write_path',
        help='path to write csv with statistical test information. \
            (default: None -> does not write results)',
        type=str,
        default=None)

    parser.add_argument(
        '--alpha',
        help='significance level below which the null hypothesis is rejected. \
            (default: 0.05)',
        type=float,
        default=0.05)

    parser.add_argument(
        '--vif_threshold',
        help='if vif greater than this value, current explanatory variable is \
            colinear with the others',
        type=int,
        default=5)

    args = parser.parse_args()

    # Load data into Dataset
    data = Dataset(data_path=args.read_path)

    # Rescale minmax normalized data
    data.rescale()

    # Flatten data
    data.flatten_3D()

    # Extract x and y data
    x, y = data.x_train_val, data.y_train_val

    # Extract timesteps data structure
    y_timesteps = [y[:, t] for t in range(y.shape[1])]

    # Extract relevant data for multistep forecast horizon y
    models, reg_results, resids, exog = fit_models(
        y_timesteps=y_timesteps, x=x)

    # print(type(reg_results))
    # print('\n'.join((dir(reg_results))))

    # # Calculate statistics
    # if args.task is None:

    # Compute all inferential statistics
    multicolinearity = compute_multicolinearity(
        exog=x, vif_threshold=args.vif_threshold)

    # autocorrelation = compute_autocorrelation(
    #     resids=resids,)

    homoskedastic = compute_homoskedasticity(
        y_timesteps=y_timesteps, x=x, alpha=args.alpha)

    # homoskedastic_and_linear = compute_linearity(
    #     resids=resids, exog=exog, alpha=args.alpha)

    # Make dataframe
    indices = ['multicolinear',  'homoskedastic']

    multicolinearity_lst = [multicolinearity, np.nan]
    # autocorrelation_lst = [np.nan, autocorrelation, np.nan]
    homoskedastic_lst = [np.nan, homoskedastic]

    df_dict = {
        f'VIFs > {args.vif_threshold}': multicolinearity_lst,
        # 'Reject Durbin-Watson H0': autocorrelation_lst,
        'Reject Goldfeld-Quandt H0': homoskedastic_lst}

    df = pd.DataFrame(df_dict)
    df.index = indices

    if args.write_path:
        df.to_csv(args.write_path, index=True)
