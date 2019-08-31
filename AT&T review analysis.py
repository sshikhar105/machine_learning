import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import re

#nltk.download('stopwords')

dataset = pd.read_csv('At&T_data.csv') 
dataset['Titles'][0]
dataset['Reviews'][0]

processed_review = []
processed_title = []


for i in range(113):
    rev = re.sub(' ', ' ',dataset['Reviews'][i])
    rev = re.sub('[^a-zA-Z]', ' ', rev)
    rev = rev.lower()
    rev = rev.split()
    rev = [ps.stem(token) for token in rev if not token in stopwords.words('english')]
    rev = ' '.join(rev)
    processed_review.append(rev)
   

for i in range(113):
    ttl = re.sub(' ', ' ',dataset['Titles'][i])
    ttl = re.sub('[^a-zA-Z]', ' ', ttl)
    ttl = rev.lower()
    ttl = rev.split()
    ttl = [ps.stem(token) for token in ttl if not token in stopwords.words('English')]
    ttl = ' '.join(ttl)
    processed_title.append(ttl)
    
y = dataset["Label"].values
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(y)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(processed_review,processed_title)
X = X.toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y)

                 
from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()
n_b.fit(X_train, y_train)
n_b.score(X, y)
n_b.score(X_test, y_test)
n_b.score(X_train, y_train)
