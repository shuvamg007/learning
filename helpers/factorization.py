import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X -= self.mean

        cov_matrix = np.cov(X.T) # check dimensions
        eig_vectors, eig_values = np.linalg.eig(cov_matrix)
        eig_vectors = eig_vectors.T

        idxs = np.argsort(eig_values)[::-1]
        sorted_vecs = eig_vectors[idxs]
        sorted_vals = eig_values[idxs]

        self.components = sorted_vecs[:self.n_components]
        
    def transform(self, X):
        X -= self.mean
        return np.dot(X, self.components.T)
