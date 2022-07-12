import math
import numpy as np
from random import gauss

n = 1000
k = 50
d = k
sigma_s = 1/k

data = np.ndarray(shape = (k*n, d+k), dtype = float, order = "F")

for i in range(k*n):
    for j in range(d):
        data[i][k+j] = gauss(0, math.sqrt(sigma_s))
        
    for j in range(k):
        if (i%k == j):
            data[i][j] = 1
        else:
            data[i][j] = 0
    
data.dump("data_{}_{}_{}_1_k.dat".format(n, k, d))
