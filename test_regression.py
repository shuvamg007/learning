from sklearn import datasets
from sklearn.model_selection import train_test_split
from models.linear_regression import LinearRegression
from helpers.metrics import get_mse

X, y = datasets.make_regression(n_samples=100, n_features=1, random_state=42, noise=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

lr = LinearRegression(lr=0.001)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

error = get_mse(y_pred, y_test)
print(f'MSE: {error}')