# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 08:59:50 2020

@author: ManavChordia
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:58:40 2020

@author: ManavChordia
"""

import os
from PIL import Image, ImageOps
import numpy as np


os.chdir('C:/Users/ManavChordia/Work/IUCAA/DI/cutouts')
files = os.listdir()

images = []
for i in files:
    pic = Image.open(i).convert('LA')
    pic = np.array(list(pic.getdata(band=0)),float)
    pic.shape = (32,32)
    #pic = pic[2:30,2:30]
    images.append(pic)
    
arr = np.array(images)
arr = arr.reshape(2006,1024)

from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize 

scaler = StandardScaler() 
arr = scaler.fit_transform(arr)
arr = normalize(arr)
import pandas as pd
arr = pd.DataFrame(arr*2550)

from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering().fit(arr)
opt = clustering.fit_predict(arr)

"""
sources = {}
for i in range(2006):
    if opt[i] not in sources:
        sources[opt[i]] = []
        sources[opt[i]].append(files[i])
    else:
        sources[opt[i]].append(files[i])
"""
        
import shutil
for i in range(0, 2006):
    if opt[i] == 0:
        shutil.copy("C:/Users/ManavChordia/Work/IUCAA/DI/cutouts/" + files[i], "C:/Users/ManavChordia/Work/IUCAA/DI/cutout_0")
    else:
        shutil.copy("C:/Users/ManavChordia/Work/IUCAA/DI/cutouts/" + files[i], "C:/Users/ManavChordia/Work/IUCAA/DI/cutout_1")
"""
import matplotlib.pyplot as plt
plt.scatter(arr[0],arr[400])
"""