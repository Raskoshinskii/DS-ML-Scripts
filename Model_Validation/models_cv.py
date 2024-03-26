import pandas as pd

from sklearn.model_selection import cross_validate, cross_val_score


def cross_validate_models(
    models, cv_type, X_train, y_train, metrics, is_aggregated=True
):
    """
    Shows models results based on certain metrics and cross-validation type.

    Parameters
    ----------
    model: list
        List of models to be evaluated (Class must be passed)
    cv_type: callable
        Type of cross-validation.
    metrics: list
        List of sklearn metrics.
    is_aggregated: bool
        Wether to return aggregated result or for each fold.

    Returns
    -------
    pd.DataFrame
        DataFrame with CV results.
    """
    res_df = pd.DataFrame()
    for model in models:
        cv_results = cross_validate(
            model, x=X_train, y=y_train, cv=cv_type, scoring=metrics
        )
        cv_results["Model"] = str(model).split("(")[
            0
        ]  # extract the name of the current model
        res_df = res_df.append(pd.DataFrame(cv_results))

    # Making the first column as a model name + drop unnecessary columns
    new_columns_order = list(res_df.columns[2:-1])
    new_columns_order.insert(0, "Model")

    # Returns score either for each fold or average estimation
    if is_aggregated:
        return res_df[new_columns_order].groupby(by="Model").mean()
    else:
        return res_df[new_columns_order]


def get_models_cross_val_score(
    models, cv_type, X_train, y_train, metric, is_aggregated=True
):
    """
    Shows models results based on certain metrics and cross-validation type.

    Parameters
    ----------
    model: list
        List of models to be evaluated (Class must be passed).
    cv_type: callable
        Type of cross-validation.
    metrics: list
        List of sklearn metrics.
    is_aggregated: bool
        Wether to return aggregated result or for each fold.

    Returns
    -------
    pd.DataFrame
        DataFrame with CV results.
    """
    res_df = pd.DataFrame()
    for model in models:
        cv_results = cross_val_score(
            model, x=X_train, y=y_train, cv=cv_type, scoring=metric
        )
        res_df.append(
            pd.DataFrame({"Model": str(model).split("(")[0], "CV-Results": cv_results})
        )

    # Returns score either for each fold or average estimation
    if is_aggregated:
        return res_df.groupby(by="Model").mean()
    else:
        return res_df
