# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:06:16 2020

@author: ManavChordia
"""

import os
from PIL import Image, ImageOps
import numpy as np


os.chdir('C:/Users/ManavChordia/Work/IUCAA/DI/cutout_0')
files = os.listdir()

images = []
for i in files:
    pic = Image.open(i).convert('LA')
    pic = np.array(list(pic.getdata(band=0)),float)
    pic.shape = (32,32)
    #pic = pic[2:30,2:30]
    images.append(pic)
    
arr = np.array(images)

"""
x_train = arr[0:1900]
x_test = arr[1900:2006]
"""
#Autoencoder code

from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D, Dropout
from keras.models import Model
from keras.models import Sequential
from keras import backend as K

model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32,32,1)))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
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


encoder = K.function([model.layers[0].input], [model.layers[4].output])
 
encoder_opt = encoder([arr])[0]
encoded_images = encoder([arr])[0].reshape(-1,16*16*4)

from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize 

scaler = StandardScaler() 
arr = scaler.fit_transform(encoded_images)
arr = normalize(arr)
import pandas as pd
arr = pd.DataFrame(arr)

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
for i in range(0, 1138):
    if opt[i] == 0:
        shutil.copy("C:/Users/ManavChordia/Work/IUCAA/DI/cutout_0/" + files[i], "C:/Users/ManavChordia/Work/IUCAA/DI/cutout_0_1")
    else:
        shutil.copy("C:/Users/ManavChordia/Work/IUCAA/DI/cutout_0/" + files[i], "C:/Users/ManavChordia/Work/IUCAA/DI/cutout_0_0")