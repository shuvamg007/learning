import numpy as np
from collections import Counter

def euclidean_dist(x1, x2):
    return np.sqrt(np.sum(x1 - x2) ** 2)

class KNN:
    """K nearest neighbour classifier
    
    Parameters
    ----------
    k: int, default=5
        Number of neighbors to maintain
    """
    def __init__(self, k=5):
        self.k = k
        
    def fit(self, X, y):
        """Builds the nearest neighbour classifier
        
        Parameters
        ----------
        X: array-like of shape (n_samples, n_feats)
            Training data to be used
        y: array-like of shape (n_samples,)
            Targets for the training data
        """
        self.X = X
        self.y = y
        
    def predict(self, X_test):
        """Predicts output for the given data
        
        Parameters
        ----------
        X_test: array-like of shape (n_samples, n_feats)
            Data to be predicted
            
        Returns
        -------
        array-like of shape (n_samples,)
            Output predicted by the classifier
        """
        return [self.predict_one(x) for x in X_test]
        
    def predict_one(self, x):
        dists = [euclidean_dist(x, x_train) for x_train in self.X]
        
        idxs = np.argsort(dists)[:self.k]
        labels = self.y[idxs]
        common_label = Counter(labels).most_common(1)[0][0]
        
        return common_label
