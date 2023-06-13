from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from helpers.metrics import get_accuracy
from models.knn import KNN

model = KNN(k=10)
iris = datasets.load_iris()

X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'Accuracy: {get_accuracy(y_test, y_pred)}')
