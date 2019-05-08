# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:34:57 2019

@author: Vishnu.Kumar1
"""

import cv2
import os
import uuid

path = os.getcwd()
folder = path + '/DataSet/'
files = os.listdir(folder)

for i in files:
    if 'Approved' in i:
        img = cv2.imread(folder + i)
        horizontal_img = cv2.flip(img, 1)
        cv2.imwrite(folder + 'Approved_' + uuid.uuid4().hex +'.jpg', horizontal_img)