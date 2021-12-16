"""Module for sklearn linear regression model"""

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor


def build_mars_mlr(n_mo_jobs=-1, **kwargs):
    """Returns sklearn multioutput linear regressor."""

    estimator = LinearRegression()
    model = MultiOutputRegressor(
        estimator=estimator, n_jobs=n_mo_jobs)
    return model
