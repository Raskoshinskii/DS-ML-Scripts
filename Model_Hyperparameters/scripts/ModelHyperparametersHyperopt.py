import numpy as np

from sklearn.model_selection import cross_val_score
from hyperopt import tpe, hp, fmin, space_eval, Trials
from hyperopt.pyll.stochastic import sample as ho_sample


class ModelHyperparametersHyperopt:
    """
    Class for hyperparameters optimizations using Hyperopt library
    CV type: cross_val_score
    
    NOTE: Delete minus in _objective for return statement when using non regression metrics 
    
    """
    
    def __init__(self, model, X_train, y_train, params_space, n_trials, cv_metric, cv_type, fit_params=None, opt_algo=tpe.suggest, seed=23):
        """
        model: callable
        
        X_train/y_train: DataFrame
        
        params_space: dict
            Hyperparameters space defined according to Hyperopt documentation 
            
        n_trials: int
            Number of iterations to find optimal hyperparameters
            
        cv_metric: str
            Name for the metric to be used according (sklearn metrics)
            
        cv_type: callable
            Cross validation type
            
        fit_params: dict 
            Additional parameters for the model
            
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
        self.fit_params = fit_params
        self.opt_algo = opt_algo
        self.seed = seed
        self.trials = Trials()
        
    def _objective(self):
        """
        Defines the objective function
            
        """
        
        self.model.set_params(**self.params_space)
        
        cv_score = cross_val_score(self.model,
                                   self.X_train,
                                   self.y_train,
                                   scoring=self.cv_metric,
                                   cv=self.cv_type,
                                   error_score='reaise',
                                   fit_params=self.fit_params,
                                   n_jobs=-1)
        
        return -cv_score.mean()
        
    def optimize(self):
        """
        Find optimal hyperparameters by minimizing the objective function
        
        """
        
        return fmin(fn=self._objective,
                    space=self.params_space,
                    algo=self.opt_algo,
                    max_evals=self.n_trials,
                    trials=self.trials,
                    rstate=np.random.RandomState(self.seed))
    
# model_hyperparameters = ModelHyperparameters(model=model_name, X_train=X_train, y_train=y_train,
#                                              params_space=params_space, n_trials=50, cv_metric='roc_auc',
#                                              cv_type=StratifiedKFold(shuffle=True, random_state=SEED))

# best_params = model_hyperparameters.optimize()

# For better found parameters retrieval use space_eval
# space_eval(params_space, best_params)