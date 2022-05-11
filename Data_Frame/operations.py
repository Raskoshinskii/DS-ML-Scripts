import pandas as pd
from functools import reduce

def pd_merge_many(lst_dfs, on):
    '''
    Merges a list of DFs one by one

    lst_dfs: list 
        List of dataframes 
    on: str
        Merging column
    
    '''
    reduce_func = lambda left,right: pd.merge(left, right, on=on)
    return reduce(reduce_func, lst_dfs)