import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_mldata
dataset = fetch_mldata('MNIST original')

X = dataset.data
y = dataset.target

some_digit = X[1234]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.3 )

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

log_reg.score(X, y)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
y_pred1 = knn.predict(X_test)

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth= 12)
dtf.fit(X_train, y_train)

dtf.score(X_train, y_train)
dtf.score(X_test, y_test)
dtf.score(X, y)

y_pred = dtf.predict(X_test)
