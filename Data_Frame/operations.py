import pandas as pd
from functools import reduce


def pd_merge_many(dfs, on):
    """
    Merges a list of DataFrames one by one.

    Parameters
    ----------
    dfs: list
        list of DataFrames.
    on: str
        Merging column.
    """
    reduce_func = lambda left, right: pd.merge(left, right, on=on)
    return reduce(reduce_func, dfs)
