#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 22:33:58 2018

@author: michal
"""
# Import libraries

# 1.Keras

from keras.datasets import cifar10
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import to_categorical

from keras_sequential_ascii import keras2ascii

# 2 Other librariess
import myplots as myp
import numpy as np


# =============================================================================
#                            # DATA OVIERVIEW
# =============================================================================


(X_train,y_train),(X_test,y_test) = cifar10.load_data()

# Define parameters
m_train = X_train.shape[0]
m_test = X_test.shape[0]
num_px = X_train[25].shape[0]
num_ch = X_train[25].shape[2]

batch_size = 32
epoch = 100
num_class = 10
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

print("\n")
print("1. Data Overwiev:\n")
print("Number of training examples : ", m_train)
print("Number of test examples : ", m_test)
print("Height/Width of each size : ", num_px)
print("Number of channels : ",num_ch)
print("Image dimensions: ", X_train[0].shape)
print("Number of classes : " , num_class,"\n")
print("Class names : " , class_names)

# plot sample images
#myp.plot_samples(X_train, y_train, names = class_names,save_fig=True)

# =============================================================================
#                            # DATA PREPROCESSING
# =============================================================================

# **** Class ****

print("\n")
print("2. Class to categorical\n")
print("Before categorical : ",y_train[0])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print("After categorical : " ,y_train[0]) 

# **** Feature scalling **** 

print("\n")
print("Before scalling : \n")
print(X_train[0])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print("\n")
print("After scalling : \n")
print(X_train[0])


# =============================================================================
#                            # Define model
# =============================================================================


from keras





















