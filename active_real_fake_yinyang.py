# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 12:51:22 2020

@author: ManavChordia
"""

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

os.chdir('C:/Users/ManavChordia/Work/IUCAA/yinyang')
files = os.listdir()
arr_yinyang = ret_arr(files)

#make x_train#########################
x_train = np.concatenate((arr_real, arr_fake, arr_yinyang))
##################################

#make y_train################# bad way dude change later
y = np.ones((270, 1))
y_0 = np.zeros((270,1))
y = np.concatenate((y, y_0, y_0), axis = 1)
y_train = y
y = np.ones((298, 1))

y_0 = np.zeros((298,1))

y = np.concatenate((y_0, y, y_0), axis = 1)
y_train = np.concatenate((y_train, y))

y = np.ones((260, 1))

y_0 = np.zeros((260, 1))

y = np.concatenate((y_0, y_0, y), axis=1)
y_train = np.concatenate((y_train, y))
######################################################


os.chdir('C:/Users/ManavChordia/Work/IUCAA/testing_1')
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
model.add(Dense(3, activation= 'softmax'))
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 1)) 
x_test = np.reshape(x_test, (len(x_test), 32, 32, 1))  


num_iter = 0
while 1: 
    print("in while")
    
    if num_iter == 0:
        print("in if 1")
        model.fit(x_train, y_train, epochs=20, verbose=1)
        y_test = model.predict(x_test)
        num_iter = num_iter + 1
        print("num _t iter = " + str(num_iter))
    
    else:
        print("in else 1")
        
        real = []
        index = []
        index_y = []
        for i in range(0, len(x_test)):
            if len(np.nonzero(y_test[i]>0.9)[0]) == 1:
                y_in = np.nonzero(y_test[i]>0.9)[0][0]
                if y_in == 0:
                    index_y.append([1., 0., 0.])
                elif y_in == 1:
                    index_y.append([0., 1, 0.])
                elif y_in == 2:
                    index_y.append([0., 0., 1])
                print(len(real))
                print(i)
                real.append(x_test[i])
                #x_test = np.delete(x_test, i-len(real), axis = 0)
            else:
                index.append(x_test[i])
        x_test = np.array(index)
        print("length of index_y = " + str(len(index_y)))
                
        if len(index_y) == 0:
            print("in if 1")
            model.fit(x_train, y_train, epochs=20, verbose=1)
            y_test = model.predict(x_test)
            
                
            print(num_iter)
            
            real = []
            index = []
            index_y = []
            for i in range(0, len(x_test)):
                if len(np.nonzero(y_test[i]>0.9)[0]) == 1:
                    y_in = np.nonzero(y_test[i]>0.9)[0][0]
                    if y_in == 0:
                        index_y.append([1., 0., 0.])
                    elif y_in == 1:
                        index_y.append([0., 1., 0.])
                    elif y_in == 2:
                        index_y.append([0., 0., 1.])
                    print(len(real))
                    print(i)
                    real.append(x_test[i])
                            #x_test = np.delete(x_test, i-len(real), axis = 0)
                else:
                    index.append(x_test[i])
                    x_test = np.array(index)

            if len(index_y) == 0:
                print("in if 2")
                break
            else:
                continue
        
        
        
        real_arr = np.array(real)
        real_arr = real_arr.astype('float32')
    
        x_train = np.concatenate((real_arr, x_train))
        index_y = np.array(index_y)
        y_train = np.concatenate((index_y, y_train))
        
        if len(x_test) <= 1:
            break
        
        model.fit(x_train, y_train, epochs=20, verbose=1)
        y_test = model.predict(x_test)
        num_iter = num_iter + 1
        print("num _t iter = " + str(num_iter))
       
