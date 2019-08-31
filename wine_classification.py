import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_wine
dataset = load_wine()

X = dataset.data
y = dataset.target

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_pred1 = log_reg.predict(X_train)
y_pred2 = log_reg.predict(X)

log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)
log_reg.score(X, y)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
cm1 = confusion_matrix(y_pred1, y_train)
cm2 = confusion_matrix(y_pred2, y)