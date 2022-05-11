import numpy as np

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Function for plotting Learning Curves
def plot_with_std(n_samples, data, **kwargs):
    mu, std = data.mean(1), data.std(1)
    lines = plt.plot(n_samples, mu, '-', **kwargs)
    plt.fill_between(n_samples, mu - std, mu + std, edgecolor='none', facecolor=lines[0].get_color(), alpha=0.2)

def plot_learning_curve(model, X_train, y_train, cv_type, scorer, train_data_split=20, shuffle=True, seed):
    """
    model: callable
        Model to fit
        
    cv_type: callable
        Type of cross-validation
        
    scorer:
        Metrci to calculate

    train_data_split: int 
    """
    train_sizes = np.linspace(0.05, 1, train_data_split)
    n_train, train_curve, val_curve = learning_curve(estimator=model, X=X_train, y=y_train,
                                                     cv=cv_type, scoring=scorer, train_sizes=train_sizes,
                                                     n_jobs=-1, shuffle=shuffle, random_state=seed)
    plt.figure(figsize=(20,8))
    plot_with_std(n_train, train_curve, label='Training Scores')
    plot_with_std(n_train, val_curve, label='Validation Scores')
    plt.xlabel('Training Set Size')
    plt.ylabel(scorer)
    plt.legend()
    plt.grid()
