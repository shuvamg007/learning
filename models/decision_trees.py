import numpy as np
from collections import Counter

class Node:
    def __init__(self, feat_idxs=None, threshold=None, left=None, right=None, *, value=None):
        self.feat_idxs = feat_idxs       # which feature was used to divide at this node
        self.threshold = threshold       # what threshold of the feature was used to divide this node
        self.left = left                 # the left tree after split
        self.right = right               # the right tree after split
        self.value = value               # value, if the node is leaf 
    
    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    """A decision tree classifier, uses entropy as criterion for splitting
    
    Parameters
    ----------
    min_samples: int, default=2
        Stopping criteria 1; min samples needed for further split
    max_depth: int, default=50
        Stopping criteria 2; max depth of tree after which splitting stops
    n_features: int
        Max features allowed in a tree; adds randomness in random forests
    """
    def __init__(self, min_samples=2, max_depth=50, n_features=None):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        
    def fit(self, X, y):
        """Builds the decision tree
        
        Parameters
        ----------
        X: array-like of shape (n_samples, n_feats)
            Training data to be used
        y: array-like of shape (n_samples,)
            Targets for the training data
        """
        n_samples, n_feats = np.shape(X)
        self.n_features = n_feats if not self.n_features else min(self.n_features, n_feats)
        
        self.root = self.expand_tree(X, y)
        
    def expand_tree(self, X, y, depth=0):
        n_samples, n_feats = np.shape(X)
        uq_labels = len(np.unique(y))
        
        # base criteria
        if uq_labels == 1 or n_samples < self.min_samples or depth >= self.max_depth:
            leaf_val = self.get_leaf_value(y)
            return Node(value=leaf_val)
        
        rand_feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        
        best_feat_idxs, best_thresh = self.best_split(X, y, rand_feat_idxs)
        
        l_idxs, r_idxs = self.split(X[:, best_feat_idxs], best_thresh)
        
        l_child = self.expand_tree(X[l_idxs, :], y[l_idxs], depth + 1)
        r_child = self.expand_tree(X[r_idxs, :], y[r_idxs], depth + 1)
        
        return Node(best_feat_idxs, best_thresh, l_child, r_child)
        
    def best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_feat_idxs, split_thresh = None, None
        
        for idx in feat_idxs:
            feature = X[:, idx]
            thresholds = np.unique(feature)
            
            for thresh in thresholds:
                gain = self.calculate_gain(thresh, y, feature)
                
                if gain > best_gain:
                    best_gain = gain
                    split_feat_idxs = idx
                    split_thresh = thresh
                    
        return split_feat_idxs, split_thresh
    
    def calculate_gain(self, thresh, y, feature):
        parent_ent = self.calculate_entropy(y)
        
        l_idxs, r_idxs = self.split(feature, thresh)
        
        if len(l_idxs) == 0 or len(r_idxs) == 0:
            return 0
        
        n = len(y)
        len_l, len_r = len(l_idxs), len(r_idxs)
        ent_l, ent_r = self.calculate_entropy(y[l_idxs]), self.calculate_entropy(y[r_idxs])
        
        gain = parent_ent - ((len_l / n) * ent_l) - ((len_r / n) * ent_r)
        
        return gain
        
    def split(self, feature, thr):
        l_idxs = np.argwhere(feature <= thr).flatten()
        r_idxs = np.argwhere(feature > thr).flatten()
        return l_idxs, r_idxs
        
    def calculate_entropy(self, y):
        hist = np.bincount(y)
        px = hist / len(y)
        return -np.sum([p * np.log(p) for p in px if p > 0])
        
    def get_leaf_value(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
        
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
        return [self.traverse_tree(x, self.root) for x in X_test]
        
        
    def traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feat_idxs] <= node.threshold:
            return self.traverse_tree(x, node.left)
        
        return self.traverse_tree(x, node.right)