# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 22:24:17 2020

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


os.chdir('C:/Users/ManavChordia/Work/IUCAA/DI/dbscan_opt_encoded/-1_0.35_-1')
files = os.listdir()

images = []
for i in files:
    pic = Image.open(i).convert('LA')
    pic = np.array(list(pic.getdata(band=0)),float)
    pic.shape = (32,32)
    pic = pic.reshape(1024)
    #pic = pic[2:30,2:30]
    images.append(pic)
    
arr = np.array(images)
#arr = arr.reshape(,1024)

from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D, Dropout
from keras.models import Model
from keras.models import Sequential
from keras import backend as K

model = Sequential()
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=(32,32,1)))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(1, kernel_size=3, padding='same', activation='relu'))

model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])
model.summary()

"""
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 1)) 
x_test = np.reshape(x_test, (len(x_test), 32, 32, 1))  
"""
arr = arr.astype('float32') / 255.
arr = np.reshape(arr, (len(arr), 32, 32, 1))

#x = np.concatenate([x_train, x_test])
model.fit(arr, arr, epochs=20, verbose=1)


encoder = K.function([model.layers[0].input], [model.layers[7].output])
 
encoder_opt = encoder([arr])[0]
encoded_images = encoder([arr])[0].reshape(-1,16*16*2)

from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize 
"""
scaler = StandardScaler() 
arr = scaler.fit_transform(arr)
arr = normalize(arr)
"""
arr = 1/(1 + np.exp(-encoded_images))
import pandas as pd
#arr = pd.DataFrame(arr*1000)

from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps = 0.2, min_samples = 50).fit(arr)
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
        shutil.copy("C:/Users/ManavChordia/Work/IUCAA/DI/cutouts/"  + j, "C:/Users/ManavChordia/Work/IUCAA/DI/dbscan_opt_encoded/-1_" + str(i))

"""
"""   
import shutil
for i in sources[-1]:
    shutil.copy("C:/Users/ManavChordia/Work/IUCAA/DI/cutouts/"  + i, "C:/Users/ManavChordia/Work/IUCAA/DI/sorted" )
"""
import matplotlib.pyplot as plt
plt.scatter(arr[0],arr[400])