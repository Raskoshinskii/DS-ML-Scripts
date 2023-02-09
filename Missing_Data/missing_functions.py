def drop_nan_columns(df):
    """
    Drops columns that have only NaN values.
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame.
        
    Returns
    -------
    pd.DataFrame
        DataFrame without NaN values.
    """
    feat_to_drop = [feature for feature in df.columns if df[feature].isna().all()]
    print(f'Dropped {len(feat_to_drop)} NaN Columns')
    df.drop(columns=feat_to_drop, axis=1, inplace=True)  