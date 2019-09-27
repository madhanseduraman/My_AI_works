# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:33:27 2019

@author: Vishnu.Kumar1
"""

import face_recognition
import os
from sklearn.externals import joblib

path = os.getcwd()
folder = path + '/DemoData/'

knownEncodings = []
knownNames = []
for i in os.listdir(folder):
    for j in os.listdir(path + '/DemoData/' + i):
        name = i
        image = face_recognition.load_image_file(path + '/DemoData/' + i + '/' + j)
        try:
            encodings = face_recognition.face_encodings(image)[0]
            knownEncodings.append(encodings)
            knownNames.append(name)
        except Exception as e:
            print(e)
            print(j)

data = {"encodings": knownEncodings, "names": knownNames}
joblib.dump(data, 'Encodings.pkl')
