import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.3 )

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

knn.score(X_train, y_train)
knn.score(X_test, y_test)
knn.score(X, y)

y_pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth= 12)
dtf.fit(X_train, y_train)

dtf.score(X_train, y_train)
dtf.score(X_test, y_test)
dtf.score(X, y)

from graphviz import Source
from sklearn import tree

graph = Source( tree.export_graphviz(dtf))
png_bytes = graph.pipe(format='png')
with open('dtree_pipe.png','wb') as f:
    f.write(png_bytes)

from IPython.display import Image
Image(png_bytes)

y_pred1 = dtf.predict(X_test)

from sklearn.svm import SVC
sv = SVC()
sv.fit(X_train, y_train)

y_pred2 = sv.predict(X_test)

sv.score(X_train,y_train)
sv.score(X_test,y_test)
sv.score(X,y)
