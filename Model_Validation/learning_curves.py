import numpy as np

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


def plot_with_std(n_samples, data, **kwargs):
    """
    Plots std in the plot.

    Parameters
    ----------
    n_samples: int
        Number of samples.
    data: pd.DataFrame.
        DataFrame with features
    **kwargs
        Other key-word arguments.

    Returns
    -------
    None
        Plots std around the line/data.
    """
    mu, std = data.mean(1), data.std(1)
    lines = plt.plot(n_samples, mu, "-", **kwargs)
    plt.fill_between(
        n_samples,
        mu - std,
        mu + std,
        edgecolor="none",
        facecolor=lines[0].get_color(),
        alpha=0.2,
    )


def plot_learning_curve(
    model, X_train, y_train, cv_type, scorer, train_data_split=20, shuffle=True, seed=23
):
    """
    Plots learning curves to see if we have
    enough observations to train a model.

    Parameters
    ----------
    model: callable
        Model to fit.
    X_train: pd.DataFrame
        Train data.
    y_train: pd.DataFrame or pd.Series
        Labels/Target vector.
    cv_type: callable
        Type of cross-validation.
    scorer:
        Metrci to calculate.
    train_data_split: int
        Number of splits for the training data
        (e.g. train_data_split=3 means that training data
        will be split into 3 folds).
    shuffle: bool
        If the train data shoudl be shuffled when before fitting.
    seed: int
        Random seed.

    Returns
    -------
    None
        Plots the learning curves.
    """
    train_sizes = np.linspace(0.05, 1, train_data_split)
    n_train, train_curve, val_curve = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv_type,
        scoring=scorer,
        train_sizes=train_sizes,
        n_jobs=-1,
        shuffle=shuffle,
        random_state=seed,
    )
    plt.figure(figsize=(20, 8))
    plot_with_std(n_train, train_curve, label="Training Scores")
    plot_with_std(n_train, val_curve, label="Validation Scores")
    plt.xlabel("Training Set Size")
    plt.ylabel(scorer)
    plt.legend()
    plt.grid()
