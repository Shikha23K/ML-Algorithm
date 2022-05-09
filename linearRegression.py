
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

n=20
X=np.arange(n)# [1,2,3,...19]
y=4*X + 3*(X**2) - 100 # this function must be unknown/hidden

# we dont hv y data so generating y[] using X; 
# there is relation b/w X 7 y is linear 
# we ll hv only data X and y we hv to find out the mapping b/w X & y

m=LinearRegression()# building a model
m.fit(X[:,np.newaxis],y) 
# X have one dimensional data;
# in scikit learn requires data in 2 dimension so adding dummy axis [:,np.newaxis]

y_predict=m.predict(X[:,np.newaxis]) #we hv model m so we predict y
plt.scatter(X,y) #Mapping b/w X and y..orginal data

#plt.plot(X,y_predict,color='red')

#  prediction:
# this plot shows the predictd y value is much distant than original; 
# means there is an error
# we dont know what mapping is b/w X,y we started fitting lines might be error occurs

#lets allow model to be little flexible..its too linear

from sklearn.preprocessing import PolynomialFeatures

#try to fit 2 degree polynomial
polyModel=PolynomialFeatures(degree=2)
#trace the feature to new feature
X_poly=polyModel.fit_transform(X[:,np.newaxis])
m.fit(X_poly,y)
y_pred_poly=m.predict(X_poly)

plt.plot(X,y_pred_poly,color='red')
plt.show()
