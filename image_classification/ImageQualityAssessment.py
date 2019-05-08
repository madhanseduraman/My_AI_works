# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:53:03 2019

@author: Vishnu.Kumar1
"""

import cv2
import os
import shutil

#import pytesseract
#from PIL import Image
#
#pytesseract.pytesseract.tesseract_cmd = "C:/Users/vishnu.kumar1/Tesseract-OCR/tesseract.exe"
path = os.getcwd()
 
def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def NotBlur():
    files = os.listdir(path + '/CroppedImages')
    notblur = []
    for i in files:
        try:
            image = cv2.resize(cv2.imread(path + '/CroppedImages/' + i), 
                               (20, 20), 
                               interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(gray)
            print ('The Blurriness Score is: ', fm)
            if fm > 10000:
                notblur.append(fm)
#                text = pytesseract.image_to_string(Image.fromarray(gray))
#                print (text)
#                if text != '':
#                    notblur.append(text)
        except Exception as e:
#            print('error', e)
            pass
    if len(notblur)>0:
        result = 'ok'
    else:
        result = 'not ok'
    shutil.rmtree(path + '/CroppedImages')
    return result
