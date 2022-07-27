
"""
This case requires trainees to develop a model for predicting fraudulent transactions for a
financial company and use insights from the model to develop an actionable plan. Data for the
case is available in CSV format having 6362620 rows and 10 columns.
Candidates can use whatever method they wish to develop their machine learning model.
Following usual model development procedures, the model would be estimated on the
calibration data and tested on the validation data. This case requires both statistical analysis and
creativity/judgment. We recommend you spend time on both fine-tuning and interpreting the
results of your machine learning model.

Candidate Expectations:

Your task is to execute the process for proactive detection of fraud while answering following
questions.
1. Data cleaning including missing values, outliers and multi-collinearity.
2. Describe your fraud detection model in elaboration.
3. How did you select variables to be included in the model?
4. Demonstrate the performance of the model by using best set of tools.
5. What are the key factors that predict fraudulent customer?
6. Do these factors make sense? If yes, How? If not, How not?
7. What kind of prevention should be adopted while company update its infrastructure?
8. Assuming these actions have been implemented, how would you determine if they work?


"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
data=pd.read_csv('Fraud.csv')

#print(data.columns)

#Transaction Distribution
print('Total Transactions are :',len(data))
print("Total no of fraud : ",len(data[data.isFraud==1]))
normalT=len(data[data.isFraud==0])
percent=round((len(data[data.isFraud==1])/normalT)*100,2)
print("percentage of frodulent transaction : ",percent)

#print(data.info())
# Checking missing data in each columns

for c in data.columns:
    flag=data[c].isnull().sum()
    if(flag>0):
        print(c)
        
    else:
        print("No missing value in : ",c)

"""
Removal the Multi-colinearity if any
   step 1 Find correlational matrix
   step 2 Plot the corrleational plot
   step 3 Check whether any two or more independent columns are highle correlated
   step 4 Drop a multi colinearity column

"""   

#step 1
cor_matrix=data.corr()
#plt.figure(figsize=(18,18))
#step 2
#sns.heatmap(cor_matrix,fmt='.1f',annot=True, cmap='Blues')
#plt.show()

#step 3
def colinearity(dataset,threshold):
    corr_column=set()
    for i in range(len(cor_matrix.columns)):
        for j in range (i):
            if abs(cor_matrix.iloc[i,j]>threshold):
                column=cor_matrix.columns[i]
                corr_column.add(column)
    return corr_column

col=colinearity(data,.8)
print(col)

#step 4

data.drop(labels=col,axis=1,inplace=True)
print(data.shape)

X=np.asarray(data.amount)
Y=np.asarray(data.isFraud)

Duplicate=data.duplicated().sum()

print("Duplicated Data : ",Duplicate)

#Data Splitting

indx=np.random.permutation(np.arange(Y.size))
nTrain=int(X.size*.85)
nTest=(indx.size-nTrain)

XTrain=X[indx[:nTrain]]
YTrain=Y[indx[:nTrain]]

XTest=X[indx[nTest:]]
YTest=Y[indx[nTest:]]


# Data preprocessing

prepro=StandardScaler()
XTrain=(prepro.fit_transform(XTrain[:,np.newaxis]))
#[: = means from 0 to end of array,np.newaxis = changes data shape from row to column]
XTest=(prepro.fit_transform(XTest[:,np.newaxis]))



"""
Modeling
Types of Regression Analysis Techniques
1. Linear Regression
2. Logistic Regression
3. Ridge Regression
4. Lasso Regression
5. Polynomial Regression
6. Bayesian Linear Regression

1    Random_forest_classifier	
2	logistic_regressor	
3	Support Vector Classifier	


1 DecisionTreeClassifier	: Decision Trees (DTs) are a non-parametric supervised
                            learning method used for classification and regression.
 The goal is to create a model that predicts the value of a target variable by 
 learning simple decision rules inferred from the data features. A tree can be
 seen as a piecewise constant approximation.



"""
model1=DecisionTreeClassifier()
model1=model1.fit(XTrain,YTrain)
prediction=model1.predict(XTest)

print(classification_report(YTest, prediction))
print("training accuracy :", model1.score(XTrain, YTrain))
print("testing accuracy :", model1.score(XTest, YTest))

plt.scatter(XTest,YTest)
plt.scatter(XTest,prediction,color='red')
plt.show()
