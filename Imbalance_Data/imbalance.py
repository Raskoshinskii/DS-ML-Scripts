import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate
from tqdm.notebook import tqdm as log_progress


def get_cv_res_class_weights(
    model, X_train, y_train, cv_type, metrics, weights=np.linspace(0.05, 0.95, 20)
):
    """
    Returns cross validation results by iteratevly
    reassigning the weights to the target classes.

    Parameters
    ----------
    model: callable
        Model to test.
    X_train: pd.DataFrame or np.array
        matrix of features
    y_train: pd.Series or np.array
        Target vector.
    cv_type: callable
        Cross-validation type.
    metrics: list
        List of sklearn metrics.
    weights: np.array
        Class weights.

    Returns
    -------
    pd.DataFrame
        DataFrame with CV results.
    """
    res_df = pd.DataFrame()
    for weight in log_progress(weights):
        cv_results = cross_validate(
            model,
            X=X_train,
            y=y_train,
            cv=cv_type,
            scoring=metrics,
            error_score="raise",
            n_jobs=-1,
            fit_params={
                "model__sample_weight": y_train.apply(
                    lambda x: weight if x == 1.0 else 1 - weight
                )
            },
        )
        cv_results["class_weight"] = weight
        res_df = res_df.append(
            pd.DataFrame(pd.DataFrame(cv_results).mean()).iloc[2:, :].T
        )
    return res_df


def get_cv_results_sampling(
    model,
    X_train,
    y_train,
    cv_type,
    metrics,
    sample_iters=20,
    sampling_type="under",
    target_labels=[-1, 1],
    random_state=SEED,
):
    """
    Returns model performance depending on
    sampling type using cross validation.

    Parameters
    ----------
    model: callable
        Model to test.
    X_train: pd.DataFrame or np.array
        Matrix of features.
    y_train: pd.DataFrame or np.array
        Target vector.
    cv_type: callable
        cross-validation type.
    metrics: list
        List of sklearn metrics.
    sampling_type: str
        under/over sampling type.
    sample_iters: int
    target_labels: list
    random_state: int
        Seed to fix.

    Returns
    -------
    pd.DataFrame
        DataFrame with CV results for different sampling size.
    """
    res_df = pd.DataFrame()

    # Define number of objects for given classes
    maj_class_size, min_class_size = y_train.value_counts()

    # Split the interval into a necessary number of elements
    n_samples = np.linspace(min_class_size, maj_class_size, sample_iters)
    n_samples = np.floor(n_samples).astype("int")  # get Int values
    # Split the samples
    X_train[y_train.name] = y_train

    x_train_pos_class = X_train[X_train[y_train.name] == max(target_labels)]
    x_train_neg_class = X_train[X_train[y_train.name] == min(target_labels)]
    # Drop target from train data
    X_train.drop(y_train.name, axis=1, inplace=True)

    if sampling_type == "under":
        for sample_size in log_progress(n_samples):
            # Sample required number of objects from majority class
            x_train_neg_under = x_train_neg_class.sample(
                sample_size, random_state=random_state
            )

            # Create train data
            x_train_under = pd.concat([x_train_neg_under, x_train_pos_class], axis=0)
            y_train = x_train_under[y_train.name]

            # Drop target from train data
            x_train_under.drop(y_train.name, axis=1, inplace=True)

            # Conduct cross-validation
            cv_results = cross_validate(
                model,
                X=x_train_under,
                y=y_train,
                cv=cv_type,
                scoring=metrics,
                error_score="raise",
                n_jobs=-1,
            )
            cv_results["Sample_Size"] = sample_size
            res_df = res_df.append(
                pd.DataFrame(pd.DataFrame(cv_results).mean()).iloc[2:, :].T
            )
        return res_df
    else:
        for sample_size in log_progress(n_samples):
            # Sample required number of objects from minority class
            x_train_pos_over = x_train_pos_class.sample(
                sample_size, replace=True, random_state=random_state
            )
            x_train_over = pd.concat([x_train_neg_class, x_train_pos_over], axis=0)
            y_train = x_train_over[y_train.name]
            x_train_over.drop(y_train.name, axis=1, inplace=True)
            cv_results = cross_validate(
                model,
                X=x_train_over,
                y=y_train,
                cv=cv_type,
                scoring=metrics,
                error_score="raise",
                n_jobs=-1,
            )
            cv_results["Sample_Size"] = sample_size
            res_df = res_df.append(
                pd.DataFrame(pd.DataFrame(cv_results).mean()).iloc[2:, :].T
            )
        return res_df
