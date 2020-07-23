# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:11:33 2020

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


x_train = arr[0:1900]
x_test = arr[1900:2006]
#Autoencoder code

from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D, Dropout
from keras.models import Model
from keras.models import Sequential
from keras import backend as K

model = Sequential()
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(32,32,1)))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(1, kernel_size=3, padding='same', activation='relu'))

model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])
model.summary()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 1)) 
x_test = np.reshape(x_test, (len(x_test), 32, 32, 1))  


model.fit(x_train, x_train, epochs=10, validation_data=(x_test, x_test), verbose=1)


encoder = K.function([model.layers[0].input], [model.layers[4].output])
 
encoder_opt = encoder([x_test])[0]
encoded_images = encoder([x_test])[0].reshape(-1,8*8*8)


"""
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

encoder = Model(input_img, encoded)
encoded_img = encoder.predict(x_test)
"""