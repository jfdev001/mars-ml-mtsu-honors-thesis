"""Module for converting sol days to datetime (in prep for missing val imputation."""

import datetime
import pandas as pd


def sol_to_datetime(
        sol_series: pd.Series,
        start_year: int = 2012,
        start_month: int = 8,
        start_day: int = 5) -> list:
    """Convert SOL to datetime format.

    Steps in the algorithm:
    (1) Get previous and current sol in sol_series.
    (2) Find the difference between the two.
    (3) Append the sum of the previous datetime
        element and a timedelta where delta
        is determined by the difference from
        step 2.
    Note: This is technically an approximation
    since SOLs are slightly longer than Earth days.

    :param sol_series: <class 'pandas.core.series.Series'> containing integers
        representing SOL days. For example, at index 0 in the series, the
        value is 1 (meaning SOL 1). Since there are skips in the data,
        the difference between the previous and current SOL value can be
        converted to a time delta and added to the rover data which is
        set at the beginning of the algorithm.
    :return:  <class 'pandas.core.series.Series'> The corrected series in
        <class 'datetime.date'> format
    """
    # Date of initial collection from https://mars.nasa.gov/msl/mission/overview/
    rover_start_date = datetime.date(
        year=start_year, month=start_month, day=start_day)

    # Build new list with for loop
    datetime_list = [rover_start_date]
    for ix in range(1, len(sol_series)):

        # Previous SOLs in list
        prev_sol = sol_series.values[ix - 1]

        # Current SOLs in list
        cur_sol = sol_series.values[ix]

        # The difference between the two SOLs
        diff = cur_sol - prev_sol

        # Appending to the datetime_list
        datetime_list.append(
            datetime_list[ix - 1] + datetime.timedelta(days=int(diff)))

    # Concatenate list to the series and return
    return datetime_list
