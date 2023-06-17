import numpy as np

class NaiveBayes:
    """Builds a Naive Bayes classifier"""

    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        
        self.mean = np.zeros((num_classes, n_features), dtype=np.float64)
        self.var = np.zeros((num_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(num_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = np.mean(X_c, axis=0)
            self.var[idx, :] = np.var(X_c, axis=0)
            self.priors[idx] = X_c.shape[0] / n_samples

    def predict(self, X_test):
        y_pred = [self.predict_one(x) for x in X_test]
        return y_pred

    def predict_one(self, x):
        predictions = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self.likelihood(x, idx)))
            posterior += prior
            predictions.append(posterior)

        return self.classes[np.argmax(predictions)]

    def likelihood(self, x, idx):
        mean = self.mean[idx]
        var = self.var[idx]

        num = np.exp(- ((x - mean) ** 2) / (2 * var) )
        den = np.sqrt(2 * np.pi * var)

        return num / den
        