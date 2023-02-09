import matplotlib.pyplot as plt

def plot_nan_ration(df, figsize=(10,5), title='Numerical Features'):
    """
    Plots a histogram with NaN ratio for the features.
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with features.
    figsize: tuple
        Figure size for the plot.
    title: str
        Plot title.
    
    Returns
    -------
    None
        Plots NaN histogram for the features.
    """
    fig = plt.figure(figsize=figsize)
    plt.hist(df.isna().sum()/df.shape[0])
    plt.xlabel('NaN Fraction')
    plt.ylabel('Number of Features')
    plt.title(title)
    plt.grid();