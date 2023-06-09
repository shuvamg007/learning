from .decision_trees import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    """Builds a Random Forest Classifier. Uses DecisionTree to build the forest

    Parameters
    ----------
    min_samples: int, default=2
        Stopping criteria 1; min samples needed for further split
    max_depth: int, default=10
        Stopping criteria 2; max depth of tree after which splitting stops
    n_features: int
        Max features allowed in a tree; adds randomness in random forests
    n_trees: int, default=5
        Number of trees to build at train time
    """
    def __init__(self, max_depth=10, n_trees=5, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.n_features = n_features
        self.min_samples_split = min_samples_split

    def sample_data(self, X, y):
        n, _ = np.shape(X)
        idxs = np.random.choice(n, n, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(self.min_samples_split, self.max_depth, self.n_features)
            X_sample, y_sample = self.sample_data(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X_test):
        predictions = np.array([tree.predict(X_test) for tree in self.trees])
        predictions = np.swapaxes(predictions, 0, 1)
        y_pred = np.array([np.argmax(np.bincount(pred)) for pred in predictions])
        
        return y_pred
