from sklearn.model_selection import GridSearchCV , RandomizedSearchCV

class ModelHyperparametersGridSearchCV:
    """
    Class for hyperparameters optimizations using GridSearchCV
    CV type: cross_val_score
    
    NOTE: Delete minus in _objective for return statement when using non regression metrics.
    """

    def __init__(self, grid_type, model, X_train, y_train, params_space, cv_metric, cv_type, seed=23):
        """
        
        Parameters
        ----------
        grid_type: callable
            GridSearchCV or RandomizedSearchCV.
        model: callable
        X_train/y_train: DataFrame
        params_space: dict
            Hyperparameters Space defined according to the Optuna documentation.
        cv_metric: str or dict (google: "Running GridSearchCV using multiple evaluation metrics") 
            Name for the metric to be used according (sklearn metrics).
        cv_type: callable
            Cross validation type.
        """
        self.grid_type = grid_type
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.params_space = params_space
        self.cv_metric = cv_metric
        self.cv_type = cv_type
        self.seed = seed
        
    def optimize(self, n_iter):
        class_name = self.grid_type.__name__
        
        if class_name.lower().startswith('rand'):
            model = self.grid_type(estimator=self.model,
                                   param_distributions=self.params_space,
                                   scoring=self.cv_metric,
                                   cv=self.cv_type,
                                   n_iter=n_iter,
                                   n_jobs=-1,
                                   random_state=self.seed)
        else:
            model = self.grid_type(estimator=self.model,
                                   param_grid=self.params_space,
                                   scoring=self.cv_metric,
                                   cv=self.cv_type,
                                   n_jobs=-1)
        
        model.fit(self.X_train, self.y_train)
        return model.best_params_