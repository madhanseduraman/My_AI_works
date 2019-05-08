# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:03:24 2019

@author: Vishnu.Kumar1
"""

import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from livelossplot.keras import PlotLossesCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report

path = os.getcwd()
folder = path + '/DataSet/'
img_width = 150
img_height = 150

train = os.listdir(folder)

print ('Starting Image Pre-Processing..')

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

X, Y = prepare_data(train)

print ('Pre-Processing Completed..')

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, 
                                                  random_state=1)
nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
batch_size = 16

print ('Starting Model Training..')

model = models.Sequential()

#model.add(layers.ZeroPadding2D((1,1),input_shape=(img_width, img_height, 3)))
model.add(layers.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
#model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

#model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(32, (3, 3)))
#model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

#model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(64, (3, 3)))
#model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
#model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
#model.add(layers.BatchNormalization())
model.add(layers.Activation('sigmoid'))
  

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

checkpointer = ModelCheckpoint(filepath='models.h5', 
                               verbose=1, save_best_only=True)

plot_losses = PlotLossesCallback()

model.fit_generator(train_generator, 
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=30, 
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    callbacks=[checkpointer, plot_losses]
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

print ('Model Training Completed..')

prediction = []
for i in X_val: 
    prediction.append(int(model.predict(np.expand_dims(i, axis = 0))[0][0]))
    
accuracy = accuracy_score(Y_val, prediction)
f1score = f1_score(Y_val, prediction, average='weighted')
precision = precision_score(Y_val, prediction, average='weighted')
recall = recall_score(Y_val, prediction, average='weighted')
roc = roc_auc_score(Y_val, prediction)
print (confusion_matrix(Y_val, prediction))
target_names = ['Rejected', 'Accepted']
print (classification_report(Y_val, prediction, target_names=target_names))

df = pd.DataFrame({'Model':['accuracy', 'f1_score', 'precision', 'recall', 
                            'roc-auc'], 'values':[accuracy, 
                                                  f1score, 
                                                  precision, 
                                                  recall, 
                                                  roc]})
df.plot.bar(x='Model', y='values', title='Model Evaluation Plot')
