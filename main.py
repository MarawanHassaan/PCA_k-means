from modules_fn import *
import time

n = 1000
k = 50
d = k
m = k
PCA = False

data = np.load("data_{}_{}_{}_1_k.dat".format(n, k, d), allow_pickle = True)

print("Shape of the data: ", data.shape)

if (PCA):
    print("\nRunning pca...", end="")
    
    t1 = time.time()
    data = pca(data, m)
    t2 = time.time()
    
    print(" done in ", t2-t1)

print("Running kmeans...")

t1 = time.time()
centers, labels, cost, i = kmeans(data, k, max_iter = 3)
t2 = time.time()

print(" done in ", t2-t1)

"""
print("Best iteration: ", i)
print("Clusters:")
for i in range(k):
    print("CLUSTER ", i, " with shape: ", data[labels==i].shape)

print("Cost: ", cost)
"""

clusters_opt = clusters_dict_opt(data, k)
clusters = clusters_dict(data, k, labels)

acc = accuracy(n, k, clusters_opt, clusters)
print("Accuracy: ", acc)
