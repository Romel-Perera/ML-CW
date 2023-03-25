import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt


data = pd.read_csv('dataset/cleaned_data.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pca = pickle.load(open('models/pca_model.pkl', 'rb'))
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

param_grid_dt = {'max_depth': [None, 5, 10, 20]}
best_accuracy = 0
best_params = {}
for max_depth in param_grid_dt['max_depth']:
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train_pca, y_train)
    y_pred = dt.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = {'max_depth': max_depth}

dt_model = DecisionTreeClassifier(max_depth=best_params['max_depth'])
dt_model.fit(X_train_pca, y_train)

y_pred = dt_model.predict(X_test_pca)
print(classification_report(y_test, y_pred))

dt_cm = confusion_matrix(y_test, y_pred)
sns.heatmap(dt_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

with open('models/dt_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)
