import numpy as np
import sys
import math
from sklearn.metrics import pairwise_distances_argmin


#function to compute euclidean distance 
def distance(p1, p2): 
    return np.sum((p1 - p2)**2)

#initialisation algorithm 
def initialize(data, k):
    centroids = np.ndarray(shape = (k, data.shape[1]))
    centroids[0, :] = (data[np.random.randint( 
            data.shape[0]), :])

    dist = [sys.maxsize]*data.shape[0]
    
    #compute remaining k - 1 centroids
    for c_id in range(k-1): 
        for i in range(data.shape[0]): 
            point = data[i, :] 
            dist[i] = min(dist[i], distance(point, centroids[c_id, :]))
        
        dist = np.array(dist)/np.sum(dist)
        next_centroid = data[np.random.choice(data.shape[0], p=dist), :]
        centroids[c_id+1, :]=next_centroid
        
    return centroids

def find_clusters(data, k, centers):
    while True:
        
        #Assign labels based on closest center
        labels = pairwise_distances_argmin(data, centers)
        
        new_centers = np.ndarray(shape = (k, data.shape[1]))

        #compute new centers by taking the mean of the points in each cluster
        for i in range(k):
            cluster_i = data[labels==i]
            new_centers[i, :] = cluster_i.mean(0)

        #Check for convergence
        if np.all(centers == new_centers):
            break
        
        centers = new_centers

    return centers, labels

def compute_cost(data, centroids, labels):
    cost = 0
    for i in range(data.shape[0]):
        point = data[i, :]
        c = labels[i]
        cost += distance(point, centroids[c, :])
    
    return cost

def pca(data, m):
    u, s, vh = np.linalg.svd(data, full_matrices=False)

    for i in range(0, s.size):
        if (i > m-1):
            s[i] = 0
            
    projected = np.dot(u, np.dot(np.diag(s), vh))

    return projected

def kmeans(data, k, max_iter = 1):
    best = (0, 0, sys.maxsize, -1)
    
    for i in range(max_iter):
        centroids = initialize(data, k)
        centers, labels = find_clusters(data, k, centroids)
        cost_i = compute_cost(data, centers, labels)
        
        if (cost_i < best[2]):
            best = (centers, labels, cost_i, i)
            
    return best

def clusters_dict(data, k, labels):
    clusters = dict()
    
    for i in range(k):
        clusters[i] = tuple(map(tuple, data[labels==i]))

    return clusters

def clusters_dict_opt(data, k):
    clusters = dict()

    for i in range(data.shape[0]):
        for j in range(k):
            if (i%k == j):
                clusters[j] = clusters.get(j, set())
                clusters[j].add(tuple(data[i, :]))
                                
    return clusters

def accuracy(n, k, clusters_opt, clusters):
    matches = 0
    
    for c_i, cluster_i in clusters_opt.items():
        max_intersection = 0
        matching_cluster = -1

        for c_j, cluster_j in clusters.items():
            cur_int = len(set(cluster_i).intersection(set(cluster_j)))

            if (cur_int > max_intersection):
                max_intersection = cur_int
                matching_cluster = c_j

        clusters.pop(matching_cluster, None)
        matches += max_intersection
        
    accuracy = matches/(n*k)

    return accuracy

