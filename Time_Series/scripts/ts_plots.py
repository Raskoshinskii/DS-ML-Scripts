import statsmodels.api as sm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.decomposition import PCA


def plot_sereis(series, x_label=None, y_label=None):
    """
    Plots timeseries.

    Parameters
    ----------
    series: pd.DataFrame or pd.Series
        Time series data.
    x_label: str
        x axis name.

    Returns
    -------
    None
        Plots time series data.
    """
    plt.figure(figsize=(20, 8))
    plt.plot(series)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)


def plot_decomposition(series, figsize=(12, 9), grid=True):
    """
    Plots timeseries decomposition (trend, seasonal, residuals).

    Parameters
    ----------
    series: pd.DataFrame
        Time series data.
    figsize: tuple
        Figure size.
    grid: bool
        Wether to plot grid or not.

    Returns
    -------
    None
        Plots timeseries decomposition.
    """
    ts_compnts = sm.tsa.seasonal_decompose(series, period=360)
    titles = ["Origianl", "Trend", "Seasonal", "Resid"]

    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=figsize)
    plt.tight_layout()

    ax[0].set_title("Time Series Decomposition")
    ax[0].plot(series)
    ax[1].plot(ts_compnts.trend)
    ax[2].plot(ts_compnts.seasonal)
    ax[3].plot(ts_compnts.resid)
    for indx, title in enumerate(titles):
        ax[indx].set_ylabel(title)
        ax[indx].grid(True)


def plot_acf_pacf(series, lags=30, figsize=(12, 7)):
    """
    Plots autocorrelation and partial autocorrelation functions.

    Parameters
    ----------
    ts: pd.Series
        Time Series data.
    lags: int
        Max number of lags to plot.
    figsize: tuple
        Figure size.

    Returns
    -------
    None
        Plots ACF/PACF graph.
    """
    plt.figure(figsize=figsize)
    ax = plt.subplot(211)
    sm.graphics.tsa.plot_acf(series.values, lags=lags, ax=ax)
    plt.grid(True)

    ax = plt.subplot(212)
    sm.graphics.tsa.plot_pacf(series.values, lags=lags, ax=ax)
    plt.grid(True)


def show_features_importances(
    model, features, n_splits, scoring, cluster_indx, target_col_name="n_trips"
):
    """
    Plots feature importance for a given cluster.

    Parameters
    ----------
    model: Class (e.g. sklearn model/pipeline)
    features: pd.DataFrame
        DataFrame with features.
    n_splits: int
        Number of splits for Time Series Cross Validation.
    scoring: str
        Metric to minimize during cross validation.
    cluster_indx: int
        Cluster index to show features importance for.
    target_col_name: str
        Target feature name.

    Returns:
    -------
    None
    """
    # Train data
    X_train = features.drop(columns=target_col_name)
    y_train = features[target_col_name]

    # Time Series CV
    cv = cross_val_score(
        model,
        X_train,
        y_train,
        cv=TimeSeriesSplit(n_splits),
        scoring=scoring,
        n_jobs=-1,
    )
    cv_mae = round(cv.mean() * (-1), 2)

    model.fit(X_train, y_train)
    coefs = pd.DataFrame(model[1].coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["Importance"] = coefs.coef.apply(np.abs)

    lasso_features = coefs["Importance"].sort_values(ascending=False)

    # Feature Importances visualization
    plt.figure(figsize=(15, 7))
    sns.barplot(lasso_features.index, lasso_features)
    plt.tight_layout()
    plt.xticks(rotation=65)
    plt.title(
        f"Feate Importances Cluster {cluster_indx} CV MAE: {cv_mae} Alpha: {model[1].alpha_}"
    )


def downsize_features_pca(df, seed):
    """
    Downsizes TS features using PCA

    Parameters
    ----------
    df: pd.DataFrame
        Time Series data.
    seed: int
        Random Seed value.

    Returns:
    -------
    pd.DataFrame
        Reduced dataframe.
    """
    pca = PCA(random_state=seed)
    pca.fit(df)
    exp_var_array = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.argwhere(exp_var_array > 0.9)[0])

    pca = PCA(n_components=n_components, random_state=seed)
    res_df = pca.fit_transform(df)
    return res_df
