# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:00:07 2019

@author: Krishna
"""

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

from keras.layers import Flatten
from keras.layers import Dense

classifier=Sequential()

classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

#classifier.add(Convolution2D(32,(3,3),activation='relu'))
#classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(units=128,activation='relu'))

classifier.add(Dense(units=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',
metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,

zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

import os
os.chdir('D:\\Study\\01_2019_DS\\01_2019_DS\\Datasets\\cnn_imgs_dataset')
training_set=train_datagen.flow_from_directory('training_set',
target_size=(64,64),
batch_size=32,
class_mode='binary')
test_set=test_datagen.flow_from_directory('test_set',
target_size=(64,64),
batch_size=32,
class_mode='binary')

history=classifier.fit_generator(training_set,steps_per_epoch=100,
nb_epoch=10,validation_data=test_set,
nb_val_samples=400)

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

test_image=image.load_img('single_prediction\cat_or_dog_1.jpg',target_size=(64,64))

x1=test_image

test_image=image.img_to_array(test_image)
test_image

test_image=np.expand_dims(test_image,axis=0)
test_image

#x=preprocess_input(x1)
x=preprocess_input(test_image)

result=classifier.predict(x)
result

if result[0][0]==1:
    print('dog')
else:
    print('cat')
    
test_image = image.load_img('single_prediction/puppy.jpg', target_size = (64, 64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
x=preprocess_input(test_image)
result=classifier.predict(x)

if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'
prediction
