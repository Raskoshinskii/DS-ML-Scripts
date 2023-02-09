import pandas as pd
import numpy as np

import scipy.stats as sts
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_crammer_values(cm):
    """
    Calculates correlation between categorical features (option 1).
    
    Parameters
    ----------
    cm: pd.DataFrame
        Confusion matrix. 
        
    Returns
    -------
    
    """
    chi2_val = sts.chi2_contingency(cm)[0]
    n_observations = cm.sum().sum()
    phi_val = chi2_val/n_observations
    n_rows, n_colmns = cm.shape
    
    # Correct phi_val, n_rows and n_colmns values
    phi_val_corr = max(0, phi_val - ((n_colmns-1)*(n_rows-1))/(n_observations-1)) 
    n_rows_corr = n_rows - ((n_rows-1)**2)/(n_observations-1)
    n_colmns_corr = n_colmns - ((n_colmns-1)**2)/(n_observations-1)
    return np.sqrt(phi_val_corr/min((n_colmns_corr-1), (n_rows_corr-1)))


def get_crammer_values(
    cat_features_df,
    target,
    correction=True,
    return_p_value=True,
    ascending=False
):
    """
    Calculate correlation between categorical features and a binary target
    (option 2, preffered). If ratio doesn't meet the conditions, the feature
    will be excluded from calculation.
    
    Parameters
    ----------
    cat_features_df: pd.DataFrame
        DataFrame with only categorical features (target must be excluded!!!).  
    target: pd.Series 
        Target feature.
    return_p_value: bool
        If True returns a tuple (Feature Name, p-value, V-Crammer Value)
        Otherwise the tuple (Feature Name, V-Crammer Value).  
    ascending: bool
        If Ture Crammer Values are sorted in a ascending order.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with features and V-Crammer values.
    """
    crammer_corrs = [] # stores the calculated correlations
    for feature in cat_features_df.columns:
        cm = pd.crosstab(cat_features_df[feature], target)
        n_observations = cat_features_df[feature].shape[0]
        n_rows, n_colmns = cm.shape
        
         # Confusion Matrix must follow some conditions before applying the method
        ratio = (np.sum((cm.loc[:, -1] < 5)) + np.sum((cm.loc[:, -1] < 5)))/cm.size
        
        if ratio <= 0.2:
            # p-values option
            if return_p_value:
                chi2_res = sts.chi2_contingency(cm, correction=correction)
                crammer_corrs.append((
                        cat_features_df[feature].name,
                        chi2_res[1], 
                        np.sqrt(chi2_res[0] / (n_observations*(min(n_rows, n_colmns)-1)))
                    ))
            else:
                chi2_val = sts.chi2_contingency(cm, correction=correction)[0]
                crammer_corrs.append((
                    cat_features_df[feature].name,
                    np.sqrt(chi2_val / (n_observations*(min(n_rows, n_colmns)-1)))
                ))
    res_df = pd.DataFrame(
        crammer_corrs,
        columns=['Feature', 'p-value', 'V_Crammer_Value']
    ).sort_values(by='V_Crammer_Value', ascending=ascending)
    return res_df


def math_exp_differences(feat_df, target, ascending=False):
    """
    Calculates correlation between numerical features and a binary target
    as avg between numerical observations of class 0 and class 1.
    
    Parameters
    ----------
    feat_df: pd.DataFrame
        DataFrame with only numerical features (target must be excluded!!!).
    target: pd.Series
        Target feature.
    ascending: bool
        If Ture, the result is sorted in ascending order.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with mean differences for numerical features of class 0 and 1. 
    """
    main_df = pd.concat([feat_df, target], axis=1) # for combining features with the target
    main_df = main_df.dropna() # DataFrame must be without NaN values because it affects the result
    mat_exp_diff = [] # stores the results
    
    for feature in feat_df.columns:
        group_means = main_df[[feature, target.name]].groupby(by=target.name).mean()
        means_diff = group_means.iloc[0, 0] - group_means.iloc[1, 0]
        mat_exp_diff.append(means_diff)
        
    df = abs(pd.DataFrame({'Corr_ME_diffs':mat_exp_diff}, index=feat_df.columns))
    return df.sort_values(by='Corr_ME_diffs', ascending=ascending)


def drop_cor_features(df, thresh=0.8):
    """
    Drops correlated features that exceed a threshold value
    using Pearson correlation.
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with features.
    thresh: float
        Cut-off value.
    
    Returns
    -------
    pd.DataFrame
        DataFrame without correlated features.
    """
    corr_matrix = df.corr()
    corr_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) >= thresh:
                col_name = corr_matrix.columns[i]
                corr_features.add(col_name)
    print('Original Number of Features: ', df.shape[1])
    print('Number of Higly Correlated Features: ', len(corr_features))
    df.drop(columns=corr_features, inplace=True)
    print('New Number of Features: ', df.shape[1])
    return df
 

def caclculate_vif(df, target=None, thresh=5):
    """
    Finds and drops correlated features using Values Inflation Factor.
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with features.
    target: str 
        Name of the target feature
        
    Returns
    -------
    pd.DataFrame
        DataFrame without correlated features.
    """
    # 1. DataFrame preparation
    if target is not None:
        df = df.drop(columns=target.columns)
    col_to_drop = df.select_dtypes(['object','datetime64[ns]']).columns
    df = df.drop(columns=col_to_drop)
    df = df.dropna(axis=0)
    # 2. VIF Calculation
    features = df.columns
    feature_idx = np.arange(len(features))
    dropping = True
    original_shape = df.shape[1]
    print('Original Number of Features: ', original_shape)
    while dropping:
        dropping = False
        current_df = df[features[feature_idx]].values
        vif_values = [
            variance_inflation_factor(current_df, idx)
            for idx in np.arange(current_df.shape[1])
        ]
        max_vif = max(vif_values)
        max_vif_loc = vif_values.index(max_vif)
        if max_vif > thresh:
            feature_idx = np.delete(feature_idx, max_vif_loc)
            dropping = True  
    new_shape = df[features[feature_idx]].shape[1]
    print('Number of Features Having Large VIF: ', original_shape - new_shape)
    print('New Number of Features:', new_shape)
    return df[features[feature_idx]]