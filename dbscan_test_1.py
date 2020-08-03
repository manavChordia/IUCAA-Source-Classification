# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:58:40 2020

@author: ManavChordia
"""

import os
from PIL import Image, ImageOps
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize 


os.chdir('C:/Users/ManavChordia/Work/IUCAA/DI/dbscan_opt/-1_')
files = os.listdir()

images = []
for i in files:
    pic = Image.open(i).convert('LA')
    pic = np.array(list(pic.getdata(band=0)),float)
    pic.shape = (32,32)
    pic = normalize(pic)
    pic = pic.reshape(1024)
    #pic = pic[2:30,2:30]
    images.append(pic)
    
arr = np.array(images)
#arr = arr.reshape(,1024)


scaler = StandardScaler() 
arr = scaler.fit_transform(arr)
arr = normalize(arr)
arr = 1/(1 + np.exp(-arr))
import pandas as pd
#arr = pd.DataFrame(arr*1000)

from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps = 0.05, min_samples = 2).fit(arr)
opt = clustering.fit_predict(arr)

sources = {}
for i in range(len(files)):
    if opt[i] not in sources:
        sources[opt[i]] = []
        sources[opt[i]].append(files[i])
    else:
        sources[opt[i]].append(files[i])
"""
import shutil
for i in sources:
    for j in sources[i]:
        print(j)
        shutil.copy("C:/Users/ManavChordia/Work/IUCAA/DI/cutouts/"  + j, "C:/Users/ManavChordia/Work/IUCAA/DI/dbscan_opt_0/" + str(i))
"""
"""
        
import shutil
for i in sources[-1]:
    shutil.copy("C:/Users/ManavChordia/Work/IUCAA/DI/cutouts/"  + i, "C:/Users/ManavChordia/Work/IUCAA/DI/sorted" )
"""
import matplotlib.pyplot as plt
plt.scatter(arr[0],arr[400])