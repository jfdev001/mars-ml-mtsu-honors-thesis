"""Module for sklearn random forest model"""

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor


def build_mars_rf(n_rf_jobs=None, n_mo_jobs=None,
                  n_estimators=100, max_depth=None,
                  min_samples_split=2, min_samples_leaf=1, max_features='auto',
                  **kwargs):
    """Returns sklearn multioutput random forest regressor."""

    rf = RandomForestRegressor(
        n_jobs=n_rf_jobs,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features)
    model = MultiOutputRegressor(estimator=rf, n_jobs=n_mo_jobs)
    return model
