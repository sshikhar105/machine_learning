import numpy as np
import matplotlib.pyplot as plt
import pandas as np

dataset = pd.read_csv('adult.csv', names = ['age', 
                                                   'workclass', 
                                                   'fnlwgt', 
                                                   'education', 
                                                   'education-num', 
                                                   'marital-status', 
                                                   'occupation', 
                                                   'relationship', 
                                                   'race', 
                                                   'gender', 
                                                   'capital-gain', 
                                                   'capital-loss', 
                                                   'hours-per-week', 
                                                   'native-country', 
                                                   'salary'], 
na_values = ' ?') 
 
 
X = dataset.iloc[:, 0:14].values 
y = dataset.iloc[:, -1].values 
 
 
from sklearn.preprocessing import Imputer 
imp = Imputer() 
X[:, [0, 2, 4, 10, 11, 12]] = imp.fit_transform(X[:, [0, 2, 4, 10, 11, 12]]) 
 
 
test = pd.DataFrame(X[:, [1, 3, 5, 6, 7, 8, 9, 13]]) 
 
 
test[0].value_counts() 
test[1].value_counts() 
test[2].value_counts() 
test[3].value_counts() 
test[4].value_counts() 
test[5].value_counts() 
test[6].value_counts() 
test[7].value_counts() 
 
 
 
 
test[0] = test[0].fillna(' Private') 
test[1] = test[1].fillna(' HS-grad') 
test[2] = test[2].fillna(' Married-civ-spouse') 
test[3] = test[3].fillna(' Prof-specialty') 
test[4] = test[4].fillna(' Husband') 
test[5] = test[5].fillna(' White') 
test[6] = test[6].fillna(' Male') 
test[7] = test[7].fillna(' United-States') 
 
 
X[:, [1, 3, 5, 6, 7, 8, 9, 13]] = test 
 
 
from sklearn.preprocessing import LabelEncoder 
lab = LabelEncoder() 
 
 
X[:, 1] = lab.fit_transform(X[:, 1].astype(str)) 
X[:, 3] = lab.fit_transform(X[:, 3]) 
X[:, 5] = lab.fit_transform(X[:, 5]) 
X[:, 6] = lab.fit_transform(X[:, 6]) 
X[:, 7] = lab.fit_transform(X[:, 7]) 
X[:, 8] = lab.fit_transform(X[:, 8]) 
X[:, 9] = lab.fit_transform(X[:, 9]) 
X[:, 13] = lab.fit_transform(X[:, 13]) 
 
 
from sklearn.preprocessing import OneHotEncoder 
one = OneHotEncoder(categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]) 
X = one.fit_transform(X) 
X = X.toarray() 
 
 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X = sc.fit_transform(X) 
y = lab.fit_transform(y) 
lab.classes_ 

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)


from sklearn.ensemble import VotingClassifier
vot = VotingClassifier([('LR', log_reg),('KNN',knn),('NB',n_b)])

vot.fit(X_train,y_train)

y_pred = vot.predict(X_test)

vot.score(X_train, y_train)
vot.score(X_test, y_test)
vot.score(X, y)

from  sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(log_reg, n_estimators= 5)
bag.fit(X_train,y_train)

y_pred1 = bag.predict(X_test)


bag.score(X_train, y_train)
bag.score(X_test, y_test)
bag.score(X, y)































