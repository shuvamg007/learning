import numpy as np
from helpers.utils import unit_step

class Perceptron:
    def __init__(self, lr=0.01, n_iters=1000):
        self.W = None
        self.b = None
        self.lr = lr
        self.n_iters = n_iters
        self.act = unit_step

    def fit(self, X, y):
        m, n = np.shape(X)

        self.W = np.zeros((1, n))
        self.b = 0

        for _ in range(self.n_iters):
            y_pred = self.act(np.dot(self.W, X.T) + self.b)
            delta = (self.lr * (y_pred - y)).T

            self.W -= np.sum(delta * X, axis=0)
            self.b -= np.sum(delta)

    def predict(self, X_test):
        return np.squeeze(self.act(np.dot(X_test, self.W.T) + self.b))
