from sklearn import datasets
from sklearn.model_selection import train_test_split
from helpers.factorization import PCA
from helpers.metrics import get_accuracy
from models.knn import KNN

def test_features(X, y):
    model = KNN(k=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return get_accuracy(y_test, y_pred)

iris = datasets.load_iris()

X, y = iris.data, iris.target

pca = PCA(2)
pca.fit(X)

X_transformed = pca.transform(X)

print(f'Original shape: {X.shape}')
print(f'Reduced features shape: {X_transformed.shape}')

# testing orignal features
print(f'Accuracy on original features: {test_features(X, y)}')

# testing transformed features
print(f'Accuracy on transformed features: {test_features(X_transformed, y)}')