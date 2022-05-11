import optuna 
from optuna import samplers

from sklearn.model_selection import cross_val_score


class ModelHyperparametersOptuna:
    """
    Class for hyperparameters optimizations using Hyperopt library
    CV type: cross_val_score
    
    NOTE: Delete minus in _objective for return statement when using non regression metrics 
    
    """

    def __init__(self, model, X_train, y_train, params_space, n_trials, cv_metric, cv_type, opt_algo='tpe', direction='maximize', seed=23):
        """
        model: callable
        
        X_train/y_train: DataFrame
        
        params_space: dict
            Hyperparameters Space defined according to the Optuna documentation
        
        n_trials: int
            Number of iterations to find optimal hyperparameters
            
        cv_metric: str
            Name for the metric to be used according (sklearn metrics)
            
        cv_type: callable
            Cross validation type
            
        opt_algo: str
            Type of an optimization algorithm
            
        direction: str
            Wether to maximize or minimize the 
            
        opt_algo: callable
            Type of an algorithm that searches in a hyperparameters space 
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.params_space = params_space
        self.n_trials = n_trials
        self.cv_metric = cv_metric
        self.cv_type = cv_type
        self.direction = direction
        self.seed = seed
        self.study = None
        
        if opt_algo == 'tpe':
            self.opt_algo = samplers.TPESampler(self.seed)
        
        
    def _objective(self, trial):
        """
        Defines the objective function
        
        """
            
        self.model.set_params(**self.params_space) 
        
        cv_score = cross_val_score(self.model, self.X_train, self.y_train,
                                   scoring=self.cv_metric, cv=self.cv_type, n_jobs=-1)
        return cv_score.mean()
    
    def optimize(self):
        self.study = optuna.create_study(sampler=self.opt_algo, direction=self.direction)
        self.study.optimize(self._objective, n_trials=self.n_trials)
        
        
# model_hyperparameters = ModelHyperparameters(model=model_name, X_train=X_train, y_train=y_train, params_space=params_space,
#                                              n_trials=50, cv_metric='roc_auc',
#                                              cv_type=StratifiedKFold(shuffle=True, random_state=SEED))

# model_hyperparameters.optimize()
# model_hyperparameters.study.best_params