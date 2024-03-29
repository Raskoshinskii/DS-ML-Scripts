import pandas as pd
import numpy as np

from category_encoders.cat_boost import CatBoostEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def plot_multidim_data(
    X_train,
    y_train,
    cat_bin_imputer=SimpleImputer(strategy="most_frequent"),
    cat_bin_encoder=CatBoostEncoder(random_state=23),
    num_imputer=SimpleImputer(strategy="mean"),
    num_scaler=StandardScaler(),
    dim_red_method=PCA(n_components=2),
    sampling_method=None,
    fig_size=(12, 6),
    title="2D Feature Space (PCA)",
):
    """
    Visualizes multidimensional by using
    different dimensionality reduction algorithms.

    Parameters
    ----------
    X_train: pd.DataFrame
        Train data.
    y_train: pd.Series
        Labels/target values.
    cat_bin_imputer: callable
        Categorical and binary features imputer
        (if NaN values are present).
    cat_bin_encoder: callable
        Categorical and binary features encoder.
    num_imputer: callable
        Numeric features imputer (if NaN values are present).
    num_scaler: callable
        Scaling method for numeric features.
    dim_red_method: callable
        Dimensionality reduction method (e.g PCA, t-SNE, ...).
    fig_size: tuple
        Size of the plot.
    title: str
        Title of the plot.

    Returns
    -------
    None
        Plots multidimensional data in 2D space.
    """
    # Features Selection
    num_features = X_train.select_dtypes(include=["float64", "int64"])
    cat_bin_features = X_train.select_dtypes(include=["object"])

    # Categorical/Binary Features Processing
    cat_bin_features = cat_bin_imputer.fit_transform(cat_bin_features)
    cat_bin_features = cat_bin_encoder.fit_transform(cat_bin_features, y_train)

    # Numerical Features Processing
    num_features = num_imputer.fit_transform(num_features)
    num_features = num_scaler.fit_transform(num_features)

    # Final Matrix of Features
    final_matrix = np.hstack((num_features, cat_bin_features))
    target_name = y_train.name

    # If there is a sampling method (over/down)
    if sampling_method:
        final_matrix, y_train = sampling_method.fit_resample(
            final_matrix, y_train.values
        )

    final_matrix_2d = pd.DataFrame(dim_red_method.fit_transform(final_matrix))
    final_matrix_2d[target_name] = y_train
    print(
        "Number of Samples (Negative Class): ",
        final_matrix_2d[final_matrix_2d[target_name] == np.unique(y_train).min()].shape[
            0
        ],
    )
    print(
        "Number of Samples (Positive Class): ",
        final_matrix_2d[final_matrix_2d[target_name] == np.unique(y_train).max()].shape[
            0
        ],
    )

    # Plotting
    plt.figure(figsize=fig_size)
    plt.scatter(
        final_matrix_2d[final_matrix_2d[target_name] == np.unique(y_train)[0]][0],
        final_matrix_2d[final_matrix_2d[target_name] == np.unique(y_train)[0]][1],
        label=np.unique(y_train)[0],
    )
    plt.scatter(
        final_matrix_2d[final_matrix_2d[target_name] == np.unique(y_train)[1]][0],
        final_matrix_2d[final_matrix_2d[target_name] == np.unique(y_train)[1]][1],
        label=np.unique(y_train)[1],
    )
    plt.xlabel("First Component")
    plt.ylabel("Second Component")
    plt.title(title)
    plt.grid()
    plt.legend()
