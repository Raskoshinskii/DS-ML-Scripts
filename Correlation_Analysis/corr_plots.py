import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns 


def plot_corr_plot(df_corrs, title=None):
    """
    Plots correlation matrix between features.
    
    Parameters
    ----------
    df_corrs: pd.DataFrame.corr()
        DataFrame with correlations between the features.
    title: str
        Plot title.
        
    Returns
    -------
    None
        Plots correlation between feattures
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(df_corrs, dtype=np.bool))
    mask = mask[1:, :-1]
    corr = df_corrs.iloc[1:,:-1].copy()
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask, 
        annot=True,
        fmt=".2f",
        linewidths=5,
        cmap=cmap,
        vmin=-1,
        vmax=1, 
        cbar_kws={"shrink": .8},
        square=True
    )
    yticks = [i.upper() for i in corr.index]
    xticks = [i.upper() for i in corr.columns]
    plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
    plt.xticks(plt.xticks()[0], labels=xticks)
    plt.title(title, loc='left', fontsize=18)
    plt.show();