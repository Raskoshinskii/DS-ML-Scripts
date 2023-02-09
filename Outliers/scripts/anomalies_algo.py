import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns 
import warnings 

from sklearn.base import BaseEstimator
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler

from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")


def get_anomalies_ratio(data):
    """
    Shows anomalies ratio.
    
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame to investigate.
    
    Returns
    -------
    None
        Prints anomalies ratio in the data.
    """
    anomalies_ratio = round((sum(data)/len(data))*100, 2)
    print(f'N Anomalies: {sum(data)}')
    print(f'Anomalies Ratio: {anomalies_ratio} %')


def get_anomalies_sigma_rule(data, col, thresh=3):
    """
    Applies sigma rule for a single feature.
    
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame to test.
    col: str
        Feature name.
    thresh: int
        Sigma value (e.g. 3/2).
    
    Returns
    -------
    pd.Series
        Boolean Series mask of outliers.
    """
    feature_mean = data[col].mean()
    feature_std = data[col].std()
    upper_bound = feature_mean + thresh * feature_std
    lower_bound = feature_mean - thresh * feature_std
    anomalies_mask = pd.concat(
        [data[col] > upper_bound, data[col] < lower_bound],
        axis=1
    ).any(1)
    return anomalies_mask, upper_bound, lower_bound 


def get_anomalies_irq(data, col, thresh):
    """
    Finds outliers/anomalies using IRQ.
    
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame to test.
    col: str
        Feature name.
    thresh: int
        Count of IRQ to apply. 
    
    Returns
    -------
    pd.Sereis 
        Boolean Series Mask of outliers.
    """
    IRQ = data[col].quantile(0.75) - data[col].quantile(0.25)
    upper_bound = data[col].quantile(0.75) + (thresh * IRQ)
    lower_bound = data[col].quantile(0.25) - (thresh * IRQ)
    anomalies_mask = pd.concat([data[col] > upper_bound, data[col] < lower_bound], axis=1).any(1)
    return anomalies_mask, upper_bound, lower_bound


def get_anomalies_summary(data, func, features=None, thresh=3):
    """
    Provides a summary table for found outliers.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame to test.
    func: callable
        Method for outliers detection (sigma rule, IRQ ...).
    features: list
        List of features.
    thresh: int
        Cut-off value.
        
    Returns
    -------
    tuple
        pd.DataFrame
            Anomalies summary DataFrame
        pd.Sereis
            Outliers Series.
    """
    if features:
        features_to_check = features
    else:
        features_to_check = data.columns 
        
    outliers = pd.Series(
        data=[False] * data.shape[0],
        index=data[features_to_check].index,
        name='is_outlier'
    )
    anomalies_summary = {}
    for feature in features_to_check:
        anomalies_mask, upper_bound, lower_bound = func(data, feature, thresh=thresh)
        anomalies_summary[feature] = [
            upper_bound,
            lower_bound,
            sum(anomalies_mask),
            100*sum(anomalies_mask)/len(anomalies_mask)
        ]
        outliers[
            anomalies_mask[anomalies_mask].index
        ] = True
    anomalies_summary = pd.DataFrame(anomalies_summary).T
    anomalies_summary.columns=[
        'upper_bound', 'lower_bound', 'anomalies_count', 'anomalies_percentage'
    ]
    anomalies_ration = round(anomalies_summary['anomalies_percentage'].sum(), 2)
    print(f'Total Outliers Ration: {anomalies_ration} %')
    return anomalies_summary, outliers 


def find_optimal_eps(
    scaled_data,
    init_eps=0.05,
    eps_inc=0.05,
    outliers_thresh=0.1
):
    """
    Finds optimal Eps parameter for DBSCAN algorithm
    that provides certain outliers ration.
    
    Parameters
    ----------
    scaled_data: pd.DataFrame
        Scaled data.
    init_eps: int
        Initial Eps value.
    eps_inc: int
        Eps increment for each new iteration.
    outliers_thresh: float
        Outliers ratio that must not be exceeded.
        
    Returns
    -------
    tuple
        list
            List containing number of clusters.
        list
            List containing anomalies ratio.
        list
            List containing eps history.
    """
    # At the beginning all objects are outliers
    anomalies_ratio = 1.0
    
    n_clusters = []
    anomalies_ratio_lst = []
    
    # Start with the smallest Eps value 
    eps = init_eps
    eps_history = [eps]
    
    # Define anomaly threshold according to outliers_thresh parameter
    while anomalies_ratio > outliers_thresh:
        model = DBSCAN(eps=eps).fit(scaled_data)
        # Get clusters (label -1 ~ means outliers/anomalies)
        labels = model.labels_ 
        # Count number of unique clusters (without anomaly cluster)
        n_clusters.append(len(np.unique(labels))-1) 

        # Count the current anomaly ratio
        is_label_anomaly = np.array([1 if label == -1 else 0 for label in labels])
        anomalies_ratio = sum(is_label_anomaly)/len(is_label_anomaly)
        anomalies_ratio_lst.append(anomalies_ratio)

        # Increase Eps value 
        eps += eps_inc
        eps_history.append(eps)
    return n_clusters, anomalies_ratio_lst, eps_history


def plot_eps_vs_anomalies_ratio(eps_history, anomalies_ratio_history):
    """
    Plots eps vs anomaly ratio plot.
    
    Parameters
    ----------
    eps_history: list 
        List with Eps values.
    anomalies_ratio_history: list 
        List with anomaly ratio values.
    
    Returns
    -------
    None 
        Plots the graph.
    """
    eps_values = eps_history[:-1]
    fig, ax1 = plt.subplots()
    
    # N_Clusters vs Eps
    ax_1_color = 'red'
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Number of Clusters', color=ax_1_color)
    ax1.plot(eps_values, n_clusters, color=ax_1_color)
    ax1.tick_params(axis='y', labelcolor=ax_1_color)

    # Define another axis on the same plot 
    ax2 = ax1.twinx() 
    
    # Anomalies Ratio vs Eps
    ax2.set_ylabel('Anomalies Ratio')  
    ax2.plot(eps_values, anomalies_ratio_history)
    fig.tight_layout()
    plt.grid(True);