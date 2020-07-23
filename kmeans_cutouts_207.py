# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:21:19 2020

@author: ManavChordia
"""

import os
from PIL import Image, ImageOps
import numpy as np


x = os.chdir('C:/Users/ManavChordia/Work/IUCAA/DI/cutouts')
files = os.listdir()
"""
for count, filename in enumerate(os.listdir("C:/Users/ManavChordia/Work/IUCAA/DI/cutouts/")): 
    dst ="cutout_" + str(count) + ".jpg"
    src ='C:/Users/ManavChordia/Work/IUCAA/DI/cutouts/'+ filename 
    dst ='C:/Users/ManavChordia/Work/IUCAA/DI/cutouts/'+ dst 
    os.rename(src, dst)

files = os.listdir()
"""

images = []
for i in files:
    pic = Image.open(i).convert('LA')
    pic = np.array(list(pic.getdata(band=0)),float)
    pic.shape = (32,32)
    pic = pic.reshape([1024])
    images.append(pic) 

from pandas import DataFrame
from sklearn.cluster import KMeans


images = np.array(images)

images = images.reshape([2006,1024])
df = DataFrame(images)

kmeans = KMeans(n_clusters = 2).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

pred = []
for x in range(0,2006):
    dist = []
    for i in range(2):
        rms = 0
        for j in range(1024):
            rms = rms + (int(df.iloc[x,j]) - int(centroids[i][j]))**2
        dist.append(rms/10)
    pred.append([min(dist), dist.index(min(dist))])
    
    
import shutil

for i in range(0, 2006):
    if pred[i][1] == 0:
        shutil.move("C:/Users/ManavChordia/Work/IUCAA/DI/cutouts/" + files[i], "C:/Users/ManavChordia/Work/IUCAA/DI/cutout_0")
    else:
        shutil.move("C:/Users/ManavChordia/Work/IUCAA/DI/cutouts/" + files[i], "C:/Users/ManavChordia/Work/IUCAA/DI/cutout_1")
      
