# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 22:22:38 2020

@author: ManavChordia
"""

import numpy as np

mnist_test = np.load('mnist_test.npy')
encoded_imgs = np.load('x_test_mnist.npy')

e = np.empty([89, 128])
for i in range(89):
    e[i] = encoded_imgs[i].flatten()


from pandas import DataFrame
from sklearn.cluster import KMeans

df = DataFrame(e)

kmeans = KMeans(n_clusters = 2).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)
centroids = centroids*10
df = df*10

dist = []
for i in range(2):
    rms = 0
    for j in range(128):
        rms = rms + (int(df.iloc[87,j]) - int(centroids[i][j]))**2
    dist.append(rms/10)
    

import matplotlib.pyplot as plt

plt.scatter(centroids[:,5],centroids[:,1])

