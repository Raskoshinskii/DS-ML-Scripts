import matplotlib.pyplot as plt
import seaborn as sns

def plot_kde_plots(
    data, n_rows, n_cols,
    figsize=(15, 20)
):
    """
    Plots KDE plots for a given dataset using Seaborn
    Note: Features must be continuous!
    
    Parameters
    ----------
    data: pd.DataFrame
        Main dataset.
    n_rows: int
        Number of rows.
        
    Returns
    -------
    None
        Plots kde plot.
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for indx, feature in enumerate(data.columns):
        sns.distplot(data[feature].dropna(), ax=axes[indx // n_cols, indx % n_cols])