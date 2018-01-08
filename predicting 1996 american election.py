# -*- coding: utf-8 -*-
"""
Created on Fri Jan 05 12:01:02 2018

@author: DELL
"""
#***************Importing the packages**********************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
sns.set()

#***************Loding the dataset***********
data=pd.read_csv('election_data_1996.csv')

print data.info()

print data.head(10)

#********************cleaning process**********************

#1.Checking the null value
null=data.isnull().sum()
print null

#There is no null value in the data

#popul:
#Description of all the features
'''    
1.popul:
Means the population in the census place in 1000

2.TVnews:
The number of time the voter views the Tv news in a week.

3.selfLR
Is the person’s self–reported political learnings from left to right. 

4.ClinLR
Is the person’s impression on Bill Clinton’s Political learning from left to right

5.DoleLR
Is the person impression of Bob Dole’s Political learnings from left to right.

6.PID
Party Identification of the person.
If the PID is
0 means the sStrong Democrat,
1 means Week democrat,
2 means Independent democrat likewise

7.age
Age of the voter.

8.educ
Education qualification of the voter.

9.income
Income of the voter.

10.vote
The vote is the target which we are going to predict using the trained logistic regression model.
vote having two possible outcomes: 0 means Clinton, 1 means Dole.
'''

train_data=data.drop('vote', axis=1)
    
#*************************Applying the corelaton matrix algorithm**************
correlation_matrix=train_data.corr()
print correlation_matrix

correlation_matrix.loc[:, :]=np.tril(correlation_matrix, k=-1)
print correlation_matrix

result=[]
already_in=set()

for col in correlation_matrix:
    perfectly_correlated=correlation_matrix[col][correlation_matrix[col]>0.6].index.tolist()
    if perfectly_correlated and col not in already_in:
        already_in.update(set(perfectly_correlated))
        print already_in
        perfectly_correlated.append(col)
        result.append(perfectly_correlated)
        
select_nested=[f[1:] for f in result]
print select_nested

select_flat=[j for i in select_nested for j in i]     
print  select_flat
    
    
#***********************Applying Variance Threshold**************
from sklearn.feature_selection import VarianceThreshold
selector=VarianceThreshold(threshold=0.8)
output=selector.fit_transform(train_data)

print output
print output.shape

#According to variance threshold the data is quite varying so no variable need to be drop
#But the result of correlation matrix tells us that selfLR need to be drop

#now lets visualise the various variables with target variable VOTE and try infer

#2.*********************visualing the columns**********************

#0:Clinton
#1:Dole
    

def plot(features):
    clinton=data[data.vote==0][features].value_counts()
    dole=data[data.vote==1][features].value_counts()
    data_frame=pd.DataFrame({'clinton':clinton, 'dole':dole})
    #data_frame.index=['dole','clinton']
    data_frame.plot.bar()

plot('popul')
plot('TVnews') 
#plot('selfLR')
plot('ClinLR')  
plot('DoleLR') 
plot('PID') 
plot('age') 
plot('educ') 
plot('income') 
    

#***************splitting into the training and test data*******************
features_drop=['popul', 'vote', 'selfLR']
X=data.drop(features_drop, axis=1)
Y=data['vote']


x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.3, random_state=25)
    
  

#********shapes*********
print x_train.shape
print x_test.shape
print y_train.shape
print y_test.shape


#********************Now applying the Logistic regression ***************
clf=LogisticRegression()
clf.fit(x_train, y_train)

prediction=clf.predict(x_test)

#*********calculating the accuracy*************
print accuracy_score(y_test, prediction)






















