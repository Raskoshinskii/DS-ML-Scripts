import pandas as pd 
import numpy as np

import statsmodels.api as sm
import warnings

from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize 
from statsmodels.tsa.holtwinters import ExponentialSmoothing 


import matplotlib.pyplot as plt
import seaborn as sns 

warnings.filterwarnings('ignore')


def plot_moving_average(
    series, window, metric,
    scale=1.96, figsize=(12, 6),
    plot_intervals=False, plot_anomalies=False
):
    """
    Plots moving average results.
    
    Parameters
    ----------
    series: pd.DataFrame
        Series.
    window: int
        Window size.
    metric: callable
        Measures an error between original series and calculated trend 
        (e.g. MSE, MAE).
    scale: float
        Sigma value (e.g. 2/3).  
    plot_intervals: bool 
        Wether to plot confidence intervals or not.
    plot_anomalies: bool 
        Wether to plot anomalies or not.
    
    Returns
    -------
    None
        Plots moving average graph for the series.
    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=figsize)
    plt.plot(rolling_mean, "g", label="Rolling Mean Trend")
    plt.title("Moving average\n Window Size = {}".format(window))

    # Confidence Intervals for Smoothed Values 
    if plot_intervals:
        error = metric(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (error + scale * deviation)
        upper_bound = rolling_mean + (error + scale * deviation)
        plt.plot(upper_bound, "r--", label='Upper Bound')
        plt.plot(lower_bound, "r--", label='Lower Bound')
        plt.fill_between(
            x=series.index,
            y1=np.squeeze(upper_bound.values),
            y2=np.squeeze(lower_bound.values),
            alpha=0.2, color="grey"
        ) 
        # Anomalies (Values that cross Confidence Intervals)
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series < lower_bound] = series[series < lower_bound]
            anomalies[series > upper_bound] = series[series > upper_bound]
            plt.plot(anomalies, "ro", markersize=10)   
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)


# Single Exponential Smoothing Algorithm 
def exponential_smoothing(series, alpha):
    """
    Exponential smoothing algorithm.
    
    Parameters
    ----------
    series: pd.DataFrame
        Series.
    alpha: float
        Smoothing value.
    
    Returns
    -------
    list
        List containing exponential smoothing values.
    """
    result = [series.iloc[0][0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(
            alpha * series.iloc[n][0] + (1 - alpha) * result[n-1]
        )
    return result


# Algorithm Visualization with different alpha values 
def plot_exponential_smoothing(series, alphas, figsize=(12, 6)):
    """
    Plots exponential smoothing using different alphas.
    
    Parameters
    ----------
    series: pd.DataFrame
        Series.
    alphas: list
        Alphas value to experiment with.
        
    Returns
    -------
    None
        Plots the graph.
    """
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=figsize)
        for alpha in alphas:
            plt.plot(exponential_smoothing(series, alpha), label=f"Alpha {alpha}")
        plt.plot(series.values, "c", label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True);


# Double Exponential Smoothing Algorithm 
def double_exponential_smoothing(series, alpha, beta):
    """
    Double exponential smoothing algorithm.
    
    Parameters
    ----------
    series: pd.DataFrame
        Series.
    alpha: float
        Smoothing parameter for level.
    beta: float
        Smoothing parameter for trend.
    """
    # First value is same as series
    result = [series.iloc[0][0]]
    for n in range(1, len(series)): # add len(series)+1 for forecasting on a single step in the future 
        if n == 1:
            level, trend = series.iloc[0][0], series.iloc[1][0] - series.iloc[0][0]
        # Forecasting
        if n >= len(series): 
            value = result[-1]
        else:
            value = series.iloc[n][0]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result


# Algorithm Visualization with different alpha and beta values 
def plot_double_exponential_smoothing(series, alphas, betas, figsize=(20, 8)):
    """
    Plots double exponential smoothing with different alphas and betas.

    Parameters
    ----------
    series: pd.DataFrame
        Series.
    alphas: list
        List of alpha values to try.
    betas: list
        List of beta values to try.
        
    Returns
    -------
    None
        Plots the graph.
    """
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=figsize)
        for alpha in alphas:
            for beta in betas:
                plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, Beta {}".format(alpha, beta))
        plt.plot(series.values, label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True);


# Plots Exponential Smoothing with certain parameters and detects anomalies 
def plot_exponential_smoothing_with_anomalies(
    series, smoothing_func,
    metric, scale=1.96,
    figsize=(12, 6),
    plot_intervals=True,
    plot_anomalies=True,
    title=None
):
    """
    Plots exponential/double exponential smoothing using certain hyperparameters combinations.
    
    Parameters
    ----------
    series: pd.DataFrame
        Series.
    smoothing_func: callable
        Functions to be used (e.g. exponential smoothing or double exponential smoothing).
    metric: callable
        Metric function.
    scale: float
        e.g. 2/3 sigma rule.
    plot_intervals: bool 
        Wether to plot confidence intervals or not.
    plot_anomalies: bool 
        Wether to plot anomalies or not.
        
    Returns
    -------
    None
        Plots the graph.
    """
    exp_smoothing_res = np.array(smoothing_func) 
    exp_smoothing_df = pd.DataFrame(exp_smoothing_res, index=series.index)  # to see the date on x axis
    
    plt.figure(figsize=figsize)
    plt.plot(exp_smoothing_df, 'g', label='Exponential Trend')        
    plt.plot(series, label="Actual Values")

    # Confidence Intervals for Smoothed Values 
    if plot_intervals:
        error = metric(series, exp_smoothing_df)
        deviation = np.std(np.squeeze(series.values) - exp_smoothing_res)
        lower_bound = exp_smoothing_df - (error + scale * deviation)
        upper_bound = exp_smoothing_df + (error + scale * deviation)
        plt.plot(upper_bound, "r--", label='Upper Bound')
        plt.plot(lower_bound, "r--", label='Lower Bound')
        plt.fill_between(
            x=series.index,
            y1=np.squeeze(upper_bound.values),
            y2=np.squeeze(lower_bound.values),
            alpha=0.2, color="grey"
        ) 
        # Anomalies (Values that cross Confidence Intervals)
        if plot_anomalies:
            # Column names must match, otherwise not working 
            upper_bound.rename(columns={0: series.columns[0]}, inplace=True)
            lower_bound.rename(columns={0: series.columns[0]}, inplace=True)
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)            
            anomalies[series < lower_bound] = series[series < lower_bound]
            anomalies[series > upper_bound] = series[series > upper_bound]
            plt.plot(anomalies, "ro", markersize=10)
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True);


# CV for Holt-Winters model 
def get_holt_cv_score(
    model_params,
    series,
    components_type,
    metric,
    season_len,
    n_splits
):
    """
    Runs Time Series CV for a given model parameters list.
    
    Parameters
    ----------
    model_params: list
        Parameters for the model (alpha, beta and gamma).
    series: pd.DataFrame
        Series.
    components_type: list
        Type of a trend (always first value) and 
        seasonal components ('additive' or 'multiplicative').
    metric: callable
        MAPE, MSE ...
    season_len: int 
        Season length. 
    cv_splits: int 
        Number of CV splits.
    
    Returns 
    -------
    np.array
        Array with CV results.
    """
    cv_errors = []
    ts_values = series.values
    trend, seasonal = components_type
    alpha, beta, gamma = model_params
    
    # Define TS Cross Validation 
    timeseries_cv = TimeSeriesSplit(n_splits=n_splits)
    
    # Iterate over folds (train and test contain list of indeces)
    for train_indxs, test_indxs in timeseries_cv.split(ts_values):
        model = ExponentialSmoothing(endog=ts_values[train_indxs], trend=trend, seasonal=seasonal, seasonal_periods=season_len)
        model = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
        actual = ts_values[test_indxs]
        preds = model.forecast(steps=len(test_indxs))
        error = metric(actual, preds)
        cv_errors.append(error)
    return np.mean(cv_errors)


def optimize_holt_hyperparams(
    model_params, series,
    components_type, metric,
    season_len, n_splits,
    show_cv_score=True
):
    """
    Find the best alpha, beta and gamma values for Holt-Winters model 
    using minimize function from scipy.optimize module and get_holt_cv_score function.
    
    Parameters
    ----------
    model_params: list
    series: DataFrame
        Series.
    components_type: list
        Type of trend (always first value) and seasonal components ('additive' or 'multiplicative')
    metric: callable
        MAPE, MSE ...
    season_len: int 
        Season length 
    cv_splits: int 
        Number of CV splits 
    Note:
        minimize returns a dictionary and the optimal hyperparameters are located in x key.
    ---
    """ 
    # Minimize the loss 
    opt_params = minimize(
        get_holt_cv_score, 
        x0=model_params,
        args=(series, components_type, metric, season_len, n_splits),
        method="TNC",
        bounds=((0, 1), (0, 1), (0, 1))
    )
    
    # Found hyperparameters
    alpha_opt, beta_opt, gamma_opt = opt_params.x
    
    print(f'Optimization Successful: {opt_params.success}')
    print(f'Optimal Alpha: {alpha_opt}\nOptimal Beta: {beta_opt}\nOptimal Gamma: {gamma_opt}\n')
    
    trend, seasonal = components_type
    model = ExponentialSmoothing(endog=series, trend=trend, seasonal=seasonal, seasonal_periods=season_len)
    model = model.fit(smoothing_level=alpha_opt, smoothing_trend=beta_opt, smoothing_seasonal=gamma_opt)

    if show_cv_score:
        cv_score = get_holt_cv_score(
            model_params=[alpha_opt, beta_opt, gamma_opt],
            series=series,
            components_type=components_type,
            metric=metric,
            season_len=season_len,
            n_splits=n_splits
        )
        print(round(cv_score, 2))
    return model


def plot_holt_forecast(
    model, train_data, metric,
    test_data=None, scale=1.96,
    figsize=(12, 6), plot_intervals=True,
    plot_anomalies=True, title=None
):
    """
    Plots Holt-Winters model forecast using already fitted model.
    
    Parameters
    ----------
    train_data: DataFrame.
    metric: callable
        Metric function.
    scale: float
        e.g. 2/3 sigma rule. 
    plot_intervals: bool 
        Wether to plot confidence intervals or not.
    plot_anomalies: bool 
        Wether to plot anomalies or not.
    ----
    """
    # Create DataFrame because train_data is a DataFrame as well 
    forecast_train = model.predict(train_data.index[0], train_data.index[-1])
    forecast_train_df = pd.DataFrame(forecast_train, index=forecast_train.index)
    
    if test_data is not None:
        forecast_test = model.predict(
            test_data.index[0], test_data.index[-1]
        )
        forecast_test_df = pd.DataFrame(
            forecast_test, index=forecast_test.index
        )
        main_df = pd.concat([forecast_train_df, forecast_test_df])
    
    plt.figure(figsize=figsize)
    plt.plot(train_data, label='Actual')        
    plt.plot(main_df, 'g', label="Model")
    plt.axvline(
        train_data.index[-1],
        ymin=0.05, ymax=0.95,
        ls='--', color='purple', lw=2.5
    )
    title = ' '.join([
        word.capitalize() for word in metric.__name__.split('_')
    ]) + ' Train' + f''
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    
    # Calculate the intervals using only train data 
    if plot_intervals:
        error = metric(train_data.values, forecast_train_df.values)
        deviation = np.std(train_data.values - forecast_train_df.values)
        lower_bound = main_df - (error + scale * deviation)
        upper_bound = main_df + (error + scale * deviation)
       
        plt.plot(lower_bound, "r--", label='Lower Bound')
        plt.plot(upper_bound, "r--", label='Upper Bound')
        plt.fill_between(
            x=main_df.index,
            y1=np.squeeze(upper_bound.values),
            y2=np.squeeze(lower_bound.values),
            alpha=0.2, color="grey"
        ) 
        title = ' '.join([
            word.capitalize() for word in metric.__name__.split('_')
        ]) + ' Train ' + f'{round(error, 2)}'
        plt.title(title)    
        
        # Anomalies (Values that cross Confidence Intervals)
        if plot_anomalies:
            # Column names must match, otherwise not working 
            upper_bound.rename(columns={0: train_data.columns[0]}, inplace=True)
            lower_bound.rename(columns={0: train_data.columns[0]}, inplace=True)
            
            # Track anomalies only for train (thus select boundaries only for train part, otherwise not working)
            upper_bound = upper_bound.loc[train_data.index]
            lower_bound = lower_bound.loc[train_data.index]
            
            anomalies = pd.DataFrame(index=train_data.index, columns=train_data.columns)            
            anomalies[train_data < lower_bound] = train_data[train_data < lower_bound]
            anomalies[train_data > upper_bound] = train_data[train_data > upper_bound]
            plt.plot(anomalies, "ro", markersize=10); 