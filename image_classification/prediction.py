# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:24:21 2019

@author: Vishnu.Kumar1
"""

import cv2
import numpy as np
from keras.models import load_model
import os
from PIL import Image
from ImageCropping import CropImage
from ImageQualityAssessment import NotBlur

path = os.getcwd()
print('Loading Deep Learning Model...')
model = load_model(path + '/models.h5')
print('Deep Learning Model Loaded..')
folders = os.listdir(path + '/ForReview')

img_width = 150
img_height = 150

def predict():
    for i in folders:
        files = os.listdir(path + '/ForReview/' + i)
        for j in files:
            try:
                height, width, channels = cv2.imread(path + '/ForReview/' + i + '/' + j).shape
                if height >= 768 and width >= 1024:
                    test_image = cv2.resize(cv2.imread(path + '/ForReview/' + i + '/' + j),
                                            (img_width, img_height), 
                                            interpolation=cv2.INTER_CUBIC)
                    test_image = np.expand_dims(test_image, axis = 0)
                    result = model.predict(test_image)
                    if result[0][0] > .5:
                        CropImage(path + '/ForReview/' + i + '/' + j)
                        notblur = NotBlur()
                        if notblur == 'ok':
                            try:
                                os.mkdir(path + '/Approved')
                            except:
                                pass
                            try:
                                os.mkdir(path + '/Approved/' + i)
                            except:
                                pass
                            im = Image.open(path + '/ForReview/' + i + '/' + j)
                            im.save(path + '/Approved/' + i + '/' + j)
                        else:
                            try:
                                os.mkdir(path + '/Rejected')
                            except:
                                pass
                            try:
                                os.mkdir(path + '/Rejected/' + i)
                            except:
                                pass
                            im = Image.open(path + '/ForReview/' + i + '/' + j)
                            im.save(path + '/Rejected/' + i + '/' + j)
                    else:
                        try:
                            os.mkdir(path + '/Rejected')
                        except:
                            pass
                        try:
                            os.mkdir(path + '/Rejected/' + i)
                        except:
                            pass
                        im = Image.open(path + '/ForReview/' + i + '/' + j)
                        im.save(path + '/Rejected/' + i + '/' + j)
                else:
                    try:
                        os.mkdir(path + '/Rejected')
                    except:
                        pass
                    try:
                        os.mkdir(path + '/Rejected/' + i)
                    except:
                        pass
                    im = Image.open(path + '/ForReview/' + i + '/' + j)
                    im.save(path + '/Rejected/' + i + '/' + j)
            except:
                pass
                
predict()