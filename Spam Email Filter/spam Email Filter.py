# -*- coding: utf-8 -*-
"""
Created on Sun Jan 07 23:40:28 2018

@author: DELL
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
########################################################################
#Reading a text-based dataset into pandas
sms=pd.read_table('https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv', header=None, names=['labels','message'])

print sms.head()

#mapping
labels_mapping={'ham':0, 'spam':1}
sms['labels']=sms['labels'].map(labels_mapping)

print sms.head(10)

X=sms['message']
Y=sms['labels']

#splitting in training and testing dataset
X_train, X_test, y_train, y_test=train_test_split(X, Y, random_state=1)

print X_train.shape

#Vectorizer
vect=CountVectorizer()

vect.fit(X_train)

X_train_dtm=vect.transform(X_train)

print X_train_dtm


X_test_dtm=vect.transform(X_test)

#modelling
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()

clf.fit(X_train_dtm, y_train)

y_pred=clf.predict(X_test_dtm)

print accuracy_score(y_test, y_pred)


print y_test.value_counts()


print len(y_test)

#using the confusion matrix
conf_matrix=confusion_matrix(y_test, y_pred)
print conf_matrix

#Accuracy
acc=(1377.0/1393)*100

#determing the false positive rate
print X_test[(y_test==0)&(y_pred==1)]

#determing the true positive rate(the are the spam)
print X_test[(y_test==1)&(y_pred==1)]

#determing the true negative rate(the are ham)
print X_test[(y_test==0)&(y_pred==0)]

