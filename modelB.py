#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 07:49:29 2018

@author: michal
"""

#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 22:33:58 2018

@author: michal
"""
# Import libraries

# 1.Keras

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization


# 2 Other librariess
import numpy as np
import pandas as pd
import myplots as myp
#import sort_images as srt
import os
import time 

# =============================================================================
#                            # DATA OVIERVIEW
# =============================================================================

(X_train,y_train),(X_test,y_test) = cifar10.load_data()
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# Define parameters
m_train = X_train.shape[0]
m_test = X_test.shape[0]
num_px = X_train[25].shape[0]
num_ch = X_train[25].shape[2]

batch_size = 128
epochs = 200
num_class = 10
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

#myp.plot_sample_img(X_train, y_train, names = class_names,save_fig=True)

# =============================================================================
#                           Preprocessing 
# =============================================================================

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# =============================================================================
#                            # Define CNN model
# =============================================================================

print("\n3. Model summary : " )


tic = time.time()



def define(model):

    model.add(Conv2D(64,(3,3),activation="relu",padding="same",input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(64,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D((2,2)))
    
    
    model.add(Conv2D(128,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Conv2D(128,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D((2,2)))
    
    model.add(Conv2D(256,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Conv2D(256,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Conv2D(256,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D((2,2)))
    
    model.add(Conv2D(512,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Conv2D(512,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Conv2D(512,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    
    return model


model = Sequential()
model = define(model)
model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



datagen = ImageDataGenerator(zoom_range=0.2,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             fill_mode="nearest",
                             horizontal_flip = True)

datagen.fit(X_train)

history2 = model.fit_generator(datagen.flow(X_train,y_train,batch_size = batch_size),
                               steps_per_epoch=len(X_train)/batch_size,epochs=epochs,
                               validation_data=(X_test,y_test),validation_steps=len(X_train)/(batch_size*2))



myp.plot_training(history2)
myp.plot_confusion_matrix(model,X_test,y_test,class_names)

loss, accuracy  = model.evaluate(X_test,y_test)

toc = time.time()
print("Exec time",str((toc-tic)) + "s")
os.system('say "Learning done"')

model.save("CIFAR-model3")



