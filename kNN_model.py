import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv('dataset/cleaned_data.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
with open('models/pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [5, 10, 15, 20],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_pca, y_train)

knn_model = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'],
                                 weights=grid_search.best_params_['weights'],
                                 metric=grid_search.best_params_['metric'])
knn_model.fit(X_train_pca, y_train)
y_pred = knn_model.predict(X_test_pca)

print(classification_report(y_test, y_pred))

knn_cm = confusion_matrix(y_test, y_pred)
sns.heatmap(knn_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

with open('models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)
