In [1]: import numpy as np

In [2]: data = np.array([[1,2], [2,3], [3,3], [4,5], [5,5]])

In [3]: np.cov(data.T)
Out[3]: 
array([[ 2.5,  2. ],
       [ 2. ,  1.8]])
