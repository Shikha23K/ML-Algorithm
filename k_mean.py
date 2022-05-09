from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

X,y=make_blobs(n_samples=500,centers=3,cluster_std=0.9)

r=np.random.permutation(np.arange(X.shape[0]))
# "permutation" process of changing the linear order of an ordered set.


mean1=X[r[0],:]
mean2=X[r[1],:]
mean3=X[r[2],:]

def createGroup(X,m1,m2,m3):
    d1=((X-m1)**2).sum(axis=1)
    d2=((X-m2)**2).sum(axis=1)
    d3=((X-m3)**2).sum(axis=1)

    #concatenate in horizontal stack
    D=np.hstack((d1[:,np.newaxis],d2[:,np.newaxis],d3[:,np.newaxis]))
    y=np.argmin(D,axis=1)
    return y

# y=createGroup(X,mean1,mean2,mean3)

indx= y==0
mean1=X[indx,:].mean(axis=0)
mean2=X[y==1,:].mean(axis=0)
mean3=X[y==2,:].mean(axis=0)

y= createGroup(X,mean1,mean2,mean3)
#compute mean for each data of dataset
plt.scatter(X[:,0],X[:,1],c=y,cmap='viridis')
plt.show()
