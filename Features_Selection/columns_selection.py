def get_bin_cat_features(df):
    """
    Returns categorical and binary feature names.
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with features.
    
    Returns
    -------
    tuple
        Binary and categorical features.
    """
    bin_columns = [
        feature for feature in df.columns if df[feature].value_counts().shape[0] == 2
    ]
    cat_columns = df.columns.difference(bin_columns).to_list()
    return bin_columns, cat_columns


def split_features(df):
    """
    Splits DataFrame into several ones 
    (binary, categorical, numeric).
    
    Parameters
    ----------
    df: pd.DataFrame 
        DataFrame with features.
    
    Returns
    -------
    tuple
        Binary, categorical and numeric features.
    """
    bin_features, cat_features = get_bin_cat_features(df)
    object_features = set(bin_features + cat_features)
    num_features = set(df.columns).difference(object_features)
    return df[num_features], df[bin_features], df[cat_features]