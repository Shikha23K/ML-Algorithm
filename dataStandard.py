# Data standardization is the process of rescaling the attributes so that
# they have mean as 0 and variance as 1

# The ultimate goal to perform standardization is to bring down all the 
# features to a common scale without distorting the differences in 
# the range of the values. 

from sklearn.preprocessing import StandardScaler
import numpy as np

X=100 * np.random.rand(200,4)+55

#center/mean of each attributes 
print(X.mean(axis=0))#axis=0 refers rows wise 

#standard deviation
print(X.std(axis=0))

s=StandardScaler()

#data standardisation
X_scaled=s.fit_transform(X)

#after standardisation mean
print(X_scaled.mean(axis=0))

#after standardisation std
print(X_scaled.std(axis=0))

