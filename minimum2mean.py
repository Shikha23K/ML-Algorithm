#minimum to mean classifier Linear regression

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X,y=make_blobs(n_samples=200,centers=2)

print(np.unique(y))
#separate out both class
indx= y==1
X1=X[indx,:]
X2=X[~indx,:]
#mean of Class I
Mean1=X1.mean(axis=0)
print("mean 1:  ",Mean1)
#mean of class II
Mean2=X2.mean(axis=0)
print("mean 2:  ",Mean2)
#random data point
indx=np.random.randint(y.size)
print("index: ",indx)
r=X[indx,:]
print("r :",r)
#Distance of random data point with mean of both class
d1=((Mean1-r)**2).sum()
d2=((Mean2-r)**2).sum()

#finding out the minimum to Mean
if(d1<d2):
    print(y[indx],1)
else:
    print(y[indx],0)
plt.scatter(X[:,0],X[:,1],c=y,cmap='viridis')
#plt.show()
