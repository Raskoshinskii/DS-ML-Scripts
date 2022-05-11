def return_bin_cat_columns(df):
    """
    Returns categorical and binary feature names
    
    """
    bin_columns = [feature for feature in df.columns if df[feature].value_counts().shape[0] == 2]
    cat_columns = df.columns.difference(bin_columns).to_list()
    
    return bin_columns, cat_columns


def get_features(dataset):
    """
    Extracts features from a dataset (e.g. binary, numeric...)
        
    """
    # Features that have object dtype
    object_features = dataset.select_dtypes(['object']).describe()
    
    # Select binary features
    bin_features = object_features.columns[object_features.loc['unique'] == 2].to_list()
    
    # Categorical features
    cat_features = object_features.columns[object_features.loc['unique'] > 2].to_list()
    
    # Numeric Features
    num_features = dataset.select_dtypes(['int64', 'float64']).columns.to_list()
    
    return dataset[num_features], dataset[bin_features], dataset[cat_features]