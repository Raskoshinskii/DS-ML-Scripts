import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def sin_transform(values):
    """
    Applies sin transformation.

    Parameters
    ----------

    values: pd.DataFrame or pd.Series
        Values to get sin harmonics for.

    Returns:
    -------
    np.array
        Sin harmonics.
    """
    return np.sin(2 * np.pi * values / len(set(values)))


def cos_transform(values):
    """
    Applies sin transformation.

    Parameters
    ----------
    values: pd.DataFrame or pd.Series
        Values to get cos harmonics for.

    Returns:
    -------
    np.array
        Cos harmonics.
    """
    return np.cos(2 * np.pi * values / len(set(values)))


def get_time_features(series, is_cyclical_encoding=False):
    """
    Extracts the main time features from time series.

    Parameters
    ----------
    series: pd.Series or pd.DataFrame
        Time Sereis Data.
    is_cyclical_encoding: bool
        Wether to apply sin/cos transformations.

    Note!!!
    ----
        Some features are commented (uncomment if needed)

    Returns:
    -------
    pd.DataFrame
        DataFrame with time features.
    """
    # In order not to change the main sereis
    features_df = series.copy()

    # Time features (continuous)
    features_df["hour"] = features_df.index.hour
    features_df["day_of_month"] = features_df.index.day
    features_df["week_of_year"] = features_df.index.weekofyear

    # Time features (categorical)
    features_df["day_name"] = features_df.index.day_name()
    features_df["month_name"] = features_df.index.month_name()

    # Take cyclicality into account
    if is_cyclical_encoding:
        features_df["sin_hour"] = sin_transform(features_df["hour"])
        features_df["cos_hour"] = cos_transform(features_df["hour"])

        features_df["sin_week_of_year"] = sin_transform(features_df["week_of_year"])
        features_df["cos_week_of_year"] = cos_transform(features_df["week_of_year"])

        # After features transfomation we no longer need previous ones
        features_df.drop(["hour", "week_of_year"], axis=1, inplace=True)

    # Weekends and Holidays (Binary)
    features_df["is_weekend"] = (
        features_df.index.weekday.isin([5, 6]) * 1
    )  # Myltiple by one to get 1/0 instead if True/False
    holidays = (
        USFederalHolidayCalendar()
        .holidays(start=features_df.index.min(), end=features_df.index.max())
        .floor("1D")
    )
    features_df["is_holiday"] = features_df.index.floor("1D").isin(holidays) * 1

    # Woorking Hour (Consider from 8:00 - 18:30 except for lunch time from 11:30-13:00)
    lunch_time = pd.date_range("12:00:00", "13:00:00", freq="h").time
    working_hours = pd.date_range("8:00:00", "19:00:00", freq="h").time
    working_hours = set(working_hours) - set(lunch_time)
    # Use parentheses
    features_df["is_working_hour"] = (
        features_df.reset_index()["tpep_pickup_datetime"].apply(
            lambda x: x.time() in working_hours
        )
        * 1
    ).values
    features_df["is_lunch_time"] = (
        features_df.reset_index()["tpep_pickup_datetime"].apply(
            lambda x: x.time() in lunch_time
        )
        * 1
    ).values

    # Stard/End of month, quarter, year
    features_df["is_month_start"] = features_df.index.is_month_start * 1
    features_df["is_month_end"] = features_df.index.is_month_end * 1
    features_df["is_quarter_start"] = features_df.index.is_quarter_start * 1
    features_df["is_quarter_end"] = features_df.index.is_quarter_end * 1
    features_df["is_year_start"] = features_df.index.is_year_start * 1
    features_df["is_year_end"] = features_df.index.is_year_end * 1

    # Season
    seasons = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
    features_df["season"] = ((features_df.index.month % 12 + 3) // 3).map(seasons)

    # Categories encoding
    ohe_columns = ["day_name"]
    features_df = pd.get_dummies(features_df, columns=ohe_columns, drop_first=True)
    return features_df


def get_lags(df, target_col_name, lag_start=1, lag_end=48, drop_target_col=True):
    """
    Computes lag features for time series.

    Parameters
    ----------
    df: pd.DataFrame
        Time Series Data.
    target_col_name: str
        Target feature name.
    lag_start: int
        Number of beginning lag.
    lag_end: int
        Number of the last lag.
    drop_target_col: bool
        Wether a target column should be dropped.

    Returns:
    -------
    pd.DataFrame
        DataFrame with lag features.
    """
    features_df = df.copy()
    for i in range(lag_start, lag_end + 1):
        features_df[f"lag_{i}"] = features_df[target_col_name].shift(i)
    features_df = features_df.dropna(axis="rows")
    if drop_target_col:
        features_df = features_df.drop(columns=target_col_name)
    return features_df


def get_harmonics(x_len, func, period, shift, factor):
    """
    Gets sin/cos harmonics.

    Parameters
    ----------
    x_len: int
        Time Series length.
    func: str
        Sin/Cos function.
    period: int
        Season length (e.g. 24, 48, 168...).
    shift: int
        Shift value (np.arrange()).
    factor: int
        Hyperparameter value.

    Returns:
    -------
    pd.DataFrame
        DataFrame with harmonics.
    """
    x = np.arange(1 + shift, x_len + shift + 1)
    if func == "sin":
        f = np.sin
    else:
        f = np.cos
    return f(x * 2.0 * np.pi * float(factor) / float(period))


def get_harmonics_df(df, func="sin"):
    """
    Gets a DataFrame with harmonic features.

    Parameters
    ----------
    df: pd.DataFrame
        Time Series data.
    func: str
        Sin/Cos function.

    Returns:
    -------
    pd.DataFrame
        DataFrame with harmonics.
    """
    features_df = pd.DataFrame()
    periods = [6, 12, 24, 168]  # use the following periods
    for period in periods:
        for shift in range(0, int(period / 2), 2):
            for factor in range(1, int(period / 2 + 1), 3):
                feature_name = "{}_{}_{}_{}".format(func, int(period), shift, factor)
                features_df[feature_name] = get_harmonics(
                    df.shape[0], func, period, shift, factor
                )
    features_df.index = df.index
    return features_df


def get_rolling_window_features(
    series,
    target_col_name,
    window_size=[12, 24],
    statistics=["avg"],
    drop_target_col=False,
):
    """
    Computes provided statistics using a rolling window.

    Parameters
    ----------
    series: pd.DataFrame or pd.Series
        Time Series data.
    target_col_name: str
        Target feature name.
    window_size: list
        Size of the rolling window.
    statistics: list
        Statistcs to calcualte.
    drop_target_col: bool
        If target column should be dropped.

    Returns:
    -------
    pd.DataFrame
        DataFrame with rolling window features.
    """
    res_df = pd.DataFrame()
    res_df["n_trips"] = series["n_trips"]
    for statistic in statistics:
        for size in window_size:
            if statistic == "avg":
                res_df[f"rolling_{statistic}_{size}"] = (
                    series[target_col_name].rolling(size).mean()
                )
            elif statistic == "min":
                res_df[f"rolling_{statistic}_{size}"] = (
                    series[target_col_name].rolling(size).min()
                )
            elif statistic == "max":
                res_df[f"rolling_{statistic}_{size}"] = (
                    series[target_col_name].rolling(size).max()
                )
            elif statistic == "sum":
                res_df[f"rolling_{statistic}_{size}"] = (
                    series[target_col_name].rolling(size).sum()
                )
            elif statistic == "std":
                res_df[f"rolling_{statistic}_{size}"] = (
                    series[target_col_name].rolling(size).std()
                )
    res_df = res_df.dropna()
    if drop_target_col:
        res_df = res_df.drop(columns=target_col_name)
    return res_df


def get_expanding_window_features(
    series,
    target_col_name,
    window_size=[12, 24],
    statistics=["avg"],
    drop_target_col=False,
):
    """
    Computes provided statistics using an expanding window.

    Parameters
    ----------
    series: pd.DataFrame or pd.Series
        Time Series data.
    target_col_name: str
        Target feature name.
    window_size: list
        Size of the rolling window.
    statistics: list
        Statistics to calcualte.
    drop_target_col: bool
        If target column should be dropped.

    Returns:
    -------
    pd.DataFrame
        DataFrame with expanding window features.
    """
    res_df = pd.DataFrame()
    res_df["n_trips"] = series["n_trips"]
    for statistic in statistics:
        for size in window_size:
            if statistic == "avg":
                res_df[f"expanding_{statistic}_{size}"] = (
                    series[target_col_name].expanding(size).mean()
                )
            elif statistic == "min":
                res_df[f"expanding_{statistic}_{size}"] = (
                    series[target_col_name].expanding(size).min()
                )
            elif statistic == "max":
                res_df[f"expanding_{statistic}_{size}"] = (
                    series[target_col_name].expanding(size).max()
                )
            elif statistic == "sum":
                res_df[f"expanding_{statistic}_{size}"] = (
                    series[target_col_name].expanding(size).sum()
                )
            elif statistic == "std":
                res_df[f"expanding_{statistic}_{size}"] = (
                    series[target_col_name].expanding(size).std()
                )
    res_df = res_df.dropna()
    if drop_target_col:
        res_df = res_df.drop(columns=target_col_name)
    return res_df
