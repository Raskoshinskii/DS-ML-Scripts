import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from tqdm.notebook import tqdm as log_progress


def show_models_cv_pipeline(models, cv_type, X_train, y_train, metrics, is_aggregated=True):
    """
    models: list 
        List of model candidates
        
    cv_type: callable
        Cross-validation type
    
    metrics: list 
        Metric to calculate
        
    is_aggregated: bool
        Wether to return aggregated results or for each fold
    """
    res_df = pd.DataFrame()
    
    for model in log_progress(models):
        model_pipeline = Pipeline([
            ('data_imputing', KNNImputer()),
            ('data_scaling', StandardScaler()),
            ('model', model)
        ])
    
        cv_results = cross_validate(model_pipeline, X=X_train, y=y_train, cv=cv_type, scoring=metrics)
        cv_results['Model'] = str(model).split('(')[0] # extract model name
        res_df = res_df.append(pd.DataFrame(cv_results))
            
    # For making the name of the model the first
    new_columns_order = list(res_df.columns[2:-1])
    new_columns_order.insert(0, 'Model')
    
    # Returning results either on each fold or aggregated 
    if is_aggregated:
        return res_df[new_columns_order].groupby(by='Model').mean()
    else:
        return res_df[new_columns_order]


def make_pipe(cat_bin_columns,
              num_columns,
              model=None,
              cat_bin_imputer=SimpleImputer(strategy='constant', fill_value='unknown'),
              cat_bin_encoder=OneHotEncoder(sparse=True, handle_unknown='ignore'),
              num_imputer=SimpleImputer(strategy='mean'),
              num_scaler=StandardScaler()):

    """
    Creates sklearn Pipeline that can process all features 

    cat_bin_columns: list 
        List with categorical and binary feature names

    num_columns: list 
        List with numeric/continuous feature names

    model: callable
        Model to use 
    ...

    """
    
    # Categorical/Binary Features Processing
    cat_bin_pipeline = Pipeline([
        ('cat_bin_imputing', cat_bin_imputer),
        ('cat_bin_encoding', cat_bin_encoder),
        ('final_imputing', SimpleImputer(strategy='constant', fill_value=0)) # Fill inf with 0
    ])
    
    # Numerical Features Processing
    num_pipeline = Pipeline([
        ('num_imputing', num_imputer),
        ('num_scaling', num_scaler)
    ])
    
    # Main Transformations
    transformations = [
        ('cat_bin_transformations', cat_bin_pipeline, cat_bin_columns),
        ('num_transformations', num_pipeline, num_columns)
    ]
    
    feature_transformations = ColumnTransformer(transformers=transformations, n_jobs=-1)
    
    main_pipeline = Pipeline([
        ('feature_transformations', feature_transformations)
    ])
    
    if model is not None:
        main_pipeline.steps.insert(1, ('model', model))
        return main_pipeline
    else:
        return main_pipeline


def show_model_cvs_pipeline_all_features(models, X_train, y_train, 
                                         cv_type, metrics,
                                         num_columns, cat_columns, bin_columns, 
                                         bin_imputer=SimpleImputer(strategy='most_frequent'), bin_encoder=OrdinalEncoder(),
                                         cat_imputer=SimpleImputer(strategy='most_frequent'), cat_encoder=OneHotEncoder(sparse=True, handle_unknown='ignore'),
                                         num_imputer=KNNImputer(), scaler=StandardScaler(),
                                         is_aggregated=True):
    """
    models: list 
        List of tested models

    cv_type: callable
        Type of cross-validation (e.g. StratifiedKFold(shuffle=True, random_state=SEED))

    metrics: list 
        List of sklearn metrics to calculate (e.g. ['precision', 'recall', 'f1', 'roc_auc'])

    is_aggregated: bool
        Wether to return the info for each fold or average estimation
    
    """
    
    # Pipeline Definition

    # Binary Features Processing 
    binary_pipeline = Pipeline([
        ('binary_imputing', bin_imputer), # Any Imputer Here
        ('binary_encoding', bin_encoder) # Any Encoder Here
    ])

    # Categorical Features Processing 
    cat_pipeline = Pipeline([
        ('cat_imputing', cat_imputer),
        ('cat_encoding', cat_encoder)
    ])

    # Numerical Features Processing
    num_pipeline = Pipeline([
        ('data_imputing', num_imputer),
        ('data_scaling', scaler)
    ])

    transformations = [
        ('num_transformations', num_pipeline, num_columns),
        ('bin_transformations', binary_pipeline, bin_columns),
        ('cat_transformations', cat_pipeline, cat_columns)
    ]

    feature_transformations = ColumnTransformer(transformers=transformations)
    
    res_df = pd.DataFrame() # CV results will be stored here
    
    # CV part for provided models
    for model in log_progress(models):
        model_pipeline = Pipeline([
            ('feature_transformations', feature_transformations),
            ('model', model)
        ])
        
        cv_results = cross_validate(model_pipeline, X=X_train, y=y_train, cv=cv_type,
                                    scoring=metrics, error_score='raise', n_jobs=-1)
        
        cv_results['Model'] = str(model).split('(')[0] # extract the name of the current model
        res_df = res_df.append(pd.DataFrame(cv_results))

    # Make the first column store the name of the model + drop unnecessary columns  
    new_columns_order = list(res_df.columns[2:-1])
    new_columns_order.insert(0, 'Model')
    
    if is_aggregated:
        return res_df[new_columns_order].groupby(by='Model').mean()
    else:
        return res_df[new_columns_order]