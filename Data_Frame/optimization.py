import pandas as pd


def optimize_floats(df):
    """
    Optimizes float values 
    
    """
    floats = df.select_dtypes(include=['float64']).columns.to_list()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df):
    """
    Optimizes int values 
    
    """
    ints = df.select_dtypes(include=['int64']).columns.to_list()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df


def optimize_objects(df, features = []):
    """
    Optimizes object values 
    
    """
    for col in df.select_dtypes(include=['object']):
        if col not in features:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_datetime(df[col])
    return df


def optimize(df, features = []):
    return optimize_floats(optimize_ints(optimize_objects(df, features)))