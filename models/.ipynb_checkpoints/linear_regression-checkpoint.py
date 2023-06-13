import numpy as np

class LinearRegression:
    """Linear regressor
    
    Parameters
    ----------
    lr: int, default=0.01
        Learning rate; controls how fast the algorithm converges
    n_iters: int, default=1000
        Number of optimization runs
    """
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        """Builds the regressor
        
        Parameters
        ----------
        X: array-like of shape (n, feats)
            Training data to be used
        y: array-like of shape (n,)
            Targets for the training data
        """
        n, feats = np.shape(X)
        self.W = np.zeros(feats)
        self.b = 0
        
        for _ in range(self.n_iters):
            y_pred = X.dot(self.W) + self.b 
            diff = (2 * (y_pred - y)) / n
            
            dW = X.T.dot(diff)
            db = np.sum(diff)
            
            self.W -= self.lr * dW
            self.b -= self.lr * db
        
    def predict(self, X_test):
        """Predicts output for the given data
        
        Parameters
        ----------
        X_test: array-like of shape (n, feats)
            Data to be predicted
            
        Returns
        -------
        array-like of shape (n,)
            Output predicted by the regressor
        """
        return X_test.dot(self.W) + self.b