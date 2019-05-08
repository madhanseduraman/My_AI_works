# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:26:22 2019

@author: Vishnu.Kumar1
"""

from keras.applications import InceptionResNetV2
import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from livelossplot.keras import PlotLossesCallback
from keras.callbacks import EarlyStopping

path = os.getcwd()
folder = path + '/DataSet/'
img_width = 150
img_height = 150

train = os.listdir(folder)

def prepare_data(list_of_images):
    x = []
    y = []
    for img in list_of_images:
        try:
            x.append(cv2.resize(cv2.imread(folder + img), 
                                (img_width, img_height), 
                                interpolation=cv2.INTER_CUBIC))
        except Exception as e:
            print(str(e))
    for i in list_of_images:
        if 'Approved' in i:
            y.append(1)
        elif 'Rejected' in i:
            y.append(0)            
    return x, y

conv_base = InceptionResNetV2(weights='imagenet', include_top=False, 
                              input_shape=(150,150,3))
X, Y = prepare_data(train)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, 
                                                  random_state=1)
nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
batch_size = 32

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) 
conv_base.trainable = False
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1. / 255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

train_generator = train_datagen.flow(np.array(X_train), 
                                     Y_train, batch_size=batch_size)

validation_generator = val_datagen.flow(np.array(X_val), 
                                        Y_val, batch_size=batch_size)

checkpointer = ModelCheckpoint(filepath='model.h5', 
                               verbose=1, save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

plot_losses = PlotLossesCallback()

model.fit_generator(train_generator, 
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=20, 
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    callbacks=[checkpointer, es, plot_losses]
                    )

d = model.history.history

plt.plot(d['acc'])
plt.plot(d['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(d['loss'])
plt.plot(d['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
