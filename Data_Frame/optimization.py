import pandas as pd


def optimize_floats(df):
    """
    Optimizes float values by downsizing them.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to be optimized.

    Returns
    -------
    pd.DataFrame
        DataFrame with downsized floats.
    """
    floats = df.select_dtypes(include=["float64"]).columns.to_list()
    df[floats] = df[floats].apply(pd.to_numeric, downcast="float")
    return df


def optimize_ints(df):
    """
    Optimizes int values by downsizing them.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to be optimized.

    Returns
    -------
    pd.DataFrame
        DataFrame with downsized ints.
    """
    ints = df.select_dtypes(include=["int64"]).columns.to_list()
    df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")
    return df


def optimize_objects(df, skip_list):
    """
    Optimizes object values downsizing them to category dtype.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to be optimized.
    skip_list: list
        list of features to be ignored.

    Returns
    -------
    pd.DataFrame
        DataFrame with downsized object columns (downsize to category).
    """
    for col in df.select_dtypes(include=["object"]):
        if col not in skip_list:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                df[col] = df[col].astype("category")
    return df


def optimize(df, skip_list):
    """
    Orchestrates functions that optimize DataFrame dtypes.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to be optimized.
    skip_list: list
        list of features to be ignored.

    Returns
    -------
    pd.DataFrame
        DataFrame with optimized datatypes.
    """
    return optimize_floats(optimize_ints(optimize_objects(df, skip_list)))
