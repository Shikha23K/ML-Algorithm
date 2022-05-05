from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('Salary_Data.csv')

X=np.asarray(df.YearsExperience)
y=np.asarray(df.Salary)

dSet=np.random.permutation(np.arange(y.size))

nTrain=int(X.size * .7)
print(dSet.size,nTrain)
nTest=dSet.size-nTrain
XTrain=X[dSet[:nTrain]]
YTrain=y[dSet[:nTrain]]
XTest=X[dSet[nTrain:]]
YTest=y[dSet[nTrain:]]

# Training the Simple Linear Regression model on the Training set
m= LinearRegression()
m.fit(XTrain[:,np.newaxis],YTrain)# modeling data

#predicting salary using Training set
Y_p_train=m.predict(XTrain[:,np.newaxis])

# Predicting the Test set results
Y_p=m.predict(XTest[:,np.newaxis])#predicting salary

# Visualising the Training set results
plt.scatter(XTrain,YTrain)
plt.plot(XTrain,Y_p_train,color='brown')
plt.show()
 
# Visualising the Test set results
plt.scatter(XTest,YTest)
plt.plot(XTest,Y_p,color='red')
plt.show()

