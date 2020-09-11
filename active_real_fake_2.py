# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:36:18 2020

@author: ManavChordia
"""

import numpy as np
import pandas as pd
import tensorflow as tf

import os
from PIL import Image, ImageOps
import numpy as np

def ret_arr(files):
    images = []
    for i in files:
        pic = Image.open(i).convert('LA')
        pic = np.array(list(pic.getdata(band=0)),float)
        pic.shape = (32,32)
        #pic = pic[2:30,2:30]
        images.append(pic)
    
    arr = np.array(images)
    return arr

os.chdir('C:/Users/ManavChordia/Work/IUCAA/Real')
files = os.listdir()

arr_real = ret_arr(files)

os.chdir('C:/Users/ManavChordia/Work/IUCAA/Fake')
files = os.listdir()

arr_fake = ret_arr(files)
x_train = np.concatenate((arr_real, arr_fake))

y_train_real = np.ones((270,1))
y_train_fake = np.zeros((298,1))
y_train = np.concatenate((y_train_real, y_train_fake))

os.chdir('C:/Users/ManavChordia/Work/IUCAA/testing')
files = os.listdir()

x_test = ret_arr(files)



#model code

from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D, Dropout, Flatten
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
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])
model.summary()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 1)) 
x_test = np.reshape(x_test, (len(x_test), 32, 32, 1))  

num_iter = 0
while 1: 
    print("in while")
    
    if num_iter == 0:
        model.fit(x_train, y_train, epochs=20, verbose=1)
        y_test = model.predict(x_test)
        num_iter = num_iter + 1
    
    else:
        print("in else 1")
        
        real = []
        index = []
        for i in range(0, len(x_test)):
            if y_test[i]>=0.90:
                print(len(real))
                print(i)
                real.append(x_test[i])
                #x_test = np.delete(x_test, i-len(real), axis = 0)
            else:
                index.append(x_test[i])
        x_test = np.array(index)
                
        if len(real) == 0:
            print("in if 1")
            model.fit(x_train, y_train, epochs=20, verbose=1)
            y_test = model.predict(x_test)
            
                
            print(num_iter)
            
            real = []
            index = []
            for i in range(0, len(x_test)):
                if y_test[i]>=0.90:
                    real.append(x_test[i])
                    #x_test = np.delete(x_test, i-len(real), axis = 0)
                else:
                    index.append(x_test[i])
            x_test = np.array(index)

            if len(real) == 0:
                print("in if 2")
                break
            else:
                continue
        
        
        
        real_arr = np.array(real)
        real_arr = real_arr.astype('float32') / 255.
    
        x_train = np.concatenate((real_arr, x_train))
        y_train = np.concatenate((np.ones((len(real_arr),1))/255., y_train))
        
        model.fit(x_train, y_train, epochs=20, verbose=1)
        y_test = model.predict(x_test)
        num_iter = num_iter + 1