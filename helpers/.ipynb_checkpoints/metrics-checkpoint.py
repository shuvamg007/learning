import numpy as np

def get_accuracy(y_test, y_pred):
    """Calculate accuracy
    
    Parameters
    ----------
    y_test : array-like of shape (n,)
        Labels from the test set
    y_pred : array-like of shape (n,)
        Model predictions
        
    Returns
    -------
    float
        Accuracy score between [0, 1]
    """
    return np.sum(y_test == y_pred) / len(y_pred)

def get_mse(y_pred, y_true):
    """Calculate mean squared error
    
    Parameters
    ----------
    y_test : array-like of shape (n,)
        Labels from the test set
    y_pred : array-like of shape (n,)
        Model predictions
        
    Returns
    -------
    float
        Mean squared error values
    """
    return np.mean((y_pred - y_true) ** 2)