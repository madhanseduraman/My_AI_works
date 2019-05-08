# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:17:17 2019

@author: Vishnu.Kumar1
"""

from TextDetection import DetectText
import os
from PIL import Image
import uuid

path = os.getcwd()

def CropImage(filename):
    img = Image.open(filename)
    coordinates = DetectText(filename)
    for ind, i in enumerate(coordinates):
        img_new = img.crop(i)
        try:
            os.mkdir(path + '/CroppedImages')
        except:
            pass
        try:
            img_new.save(path + '/CroppedImages/' + 'crop_' + uuid.uuid4().hex + '.jpg')
        except:
            pass
