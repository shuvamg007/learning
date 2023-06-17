from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from helpers.metrics import get_accuracy
from models.logistic_regression import LogisticRegression
from models.decision_trees import DecisionTree 
from models.random_forest import RandomForest
from models.naive_bayes import NaiveBayes

# model = LogisticRegression()
# model = DecisionTree()
# model = RandomForest()
model = NaiveBayes()
b_cancer = datasets.load_breast_cancer()

X, y = b_cancer.data, b_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_test, y_pred)

print(f'Accuracy: {get_accuracy(y_test, y_pred)}')
