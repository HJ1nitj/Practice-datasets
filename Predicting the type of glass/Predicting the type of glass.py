# -*- coding: utf-8 -*-


#Importing the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
sns.set()

#Loading the dataset

glass_data=pd.read_csv('glass_data.csv')


#features description
'''
1. Id number: 1 to 214
2. RI: refractive index
3. Na: Sodium (unit measurement: weight percent in the corresponding oxide, as attributes 4-10)
4. Mg: Magnesium
5. Al: Aluminum
6. Si: Silicon
7. K: Potassium
8. Ca: Calcium
9. Ba: Barium
10. Fe: Iron
'''

print glass_data.info()
print glass_data.head(10)
print glass_data.describe()

#drawing the density graph of each predictor with the all the target class
def density_plot(feature):
    x=glass_data['glass-type']
    y=glass_data[feature]
    plt.scatter(glass_data.index, y, c=x , marker='o', cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.xlabel('observation')
    plt.ylabel('weight percentage')
    #plt.show()

for i in  header:
    density_plot(i)
    plt.title(i)
    plt.show()
    


#*****************************Applying Variance threshold****************************
from sklearn.feature_selection import VarianceThreshold
selector=VarianceThreshold(threshold=0.5)
output=selector.fit_transform(glass_data)
print output.shape


#from variance threshold it is clear that there not is not much variation in the data

#Now let's implement the correleation matrix algo to get the more insights about the feature selection

#***********************CORRELATION MATRIX ALGORITHM (0.5) *********************
app_data=glass_data.drop(['glass-type'], axis=1)
data_comatrix=app_data.corr()
data_comatrix.loc[:, :]=np.tril(data_comatrix, k=-1)

result=[]
already_in=set()

for col in data_comatrix: 
    perfect_correlation=data_comatrix[col][data_comatrix[col]>=0.5].index.tolist()
    if perfect_correlation and col not in already_in:
        already_in.update(set(perfect_correlation))
        perfect_correlation.append(col)
        result.append(perfect_correlation)
        
print result

select=[i[1:] for i in result]

#['Ca', 'RI']

#Since from correlation matrix algo we are getting Ca and RI to be removed
#and from variance threshold we are getting we are getting only 4 features 
#so let's remove Ca and RI, Ba(As it is having very variation) from the dataset

#********************splitting into training and testing data**************
X=glass_data.drop(['Ca', 'RI', 'Ba' , 'glass-type'], axis=1)
Y=glass_data['glass-type']
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.3, random_state=25)


#******************Implementing the multinomial logistic regression************
clf=linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
clf.fit(x_train, y_train)

predict=clf.predict(x_test)

print accuracy_score(y_test, predict)



        



























