import numpy as np
import pandas as pd
import pydotplus
from sklearn.tree import export_graphviz
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from io import StringIO
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("C:/Users/PC/OneDrive/Desktop/AML/DT and SVM/loan_approval_dataset.csv")

df.drop('loan_id', axis=1, inplace=True)

df.columns = df.columns.str.strip()

df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

df["self_employed"] = df["self_employed"].apply(lambda x: 1 if x == "Yes" else 0)
df["education"] = df["education"].apply(lambda x: 1 if x == "Graduate" else 0)
df["loan_status"] = df["loan_status"].apply(lambda x: 1 if x == "Approved" else 0)

x = df.drop(['loan_status'], axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svc = SVC(kernel="linear")
svc.fit(X_train_scaled, y_train)

y_pred = svc.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

param_grid = {
    'C': [1, 10, 100],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf', 'linear', 'sigmoid']
}

grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

best_params_grid = grid_search.best_params_
print("Best hyperparameters:", best_params_grid)

best_svm_grid = SVC(**best_params_grid)
best_svm_grid.fit(X_train_scaled, y_train)

accuracy = best_svm_grid.score(X_test_scaled, y_test)
print("Accuracy on test set:", accuracy)

param_distributions = {
    'C': loguniform(1e-4, 1e4),
    'gamma': loguniform(1e-4, 1e4),
    'kernel': ['rbf', 'sigmoid']
}

random_search = RandomizedSearchCV(estimator=svc, param_distributions=param_distributions, n_iter=100,
                                   cv=5, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train_scaled, y_train)

best_params_rdm = random_search.best_params_
print("Best hyperparameters:", best_params_rdm)

best_svc_rdm = SVC(**best_params_rdm)
best_svc_rdm.fit(X_train_scaled, y_train)

accuracy = best_svc_rdm.score(X_test_scaled, y_test)
print("Accuracy on test set:", accuracy)

def M_S(y_real, y_pred, label):
    return pd.Series({'accuracy':accuracy_score(y_real, y_pred),
                      'precision': precision_score(y_real, y_pred,),
                      'recall': recall_score(y_real, y_pred,),
                      'f1': f1_score(y_real, y_pred,)},
                      name=label)

y_train_pred = svc.predict(X_train_scaled)
y_test_pred = svc.predict(X_test_scaled)

train_test_full_error = pd.concat([M_S(y_train, y_train_pred, 'train'),M_S(y_test, y_test_pred, 'test')],axis=1)
train_test_full_error

y_train_pred = best_svm_grid.predict(X_train_scaled)
y_test_pred = best_svm_grid.predict(X_test_scaled)

train_test_full_error = pd.concat([M_S(y_train, y_train_pred, 'train'),M_S(y_test, y_test_pred, 'test')],axis=1)
train_test_full_error

y_train_pred = best_svc_rdm.predict(X_train_scaled)
y_test_pred = best_svc_rdm.predict(X_test_scaled)

train_test_full_error = pd.concat([M_S(y_train, y_train_pred, 'train'),M_S(y_test, y_test_pred, 'test')],axis=1)
train_test_full_error
