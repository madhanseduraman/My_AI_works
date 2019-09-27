# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:29:43 2019

@author: Vishnu.Kumar1
"""

import face_recognition
import time
import cv2
import numpy as np
from sklearn.externals import joblib
import os
from Email import email, FileEmail, Template
from datetime import datetime
import pandas as pd
from PIL import Image

path = os.getcwd()

fvs = cv2.VideoCapture(0)
time.sleep(0.5)
data = joblib.load('Encodings.pkl')

def FaceRecognition():
    face_locations = []
    face_encodings = []
    face_names = []
    name_array = []
    unknown_array = []
    process_this_frame = True
    
    out = cv2.VideoWriter('output.mp4', -1, 20.0, (640,480))
    capture_duration = 7200
    start_time = time.time()
    
    while True:
        ret, frame = fvs.read()
        if frame is None:
            break
        out.write(frame)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, 
                                                             face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(data['encodings'], 
                                                         face_encoding, 
                                                         tolerance=0.45)
                name = "Unknown_Unknown"
                face_distances = face_recognition.face_distance(data['encodings'], 
                                                                face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = data['names'][best_match_index]
                face_names.append(name)
                if name == "Unknown_Unknown":
                    unknown_array.append(name)
                else:
                    dict_name = {}
                    dict_name['name'] = name
                    dict_name['Name'] = name.split('_')[0]
                    dict_name['GPN'] = name.split('_')[1]
                    dict_name['Email ID'] = name.split('_')[2]
                    dict_name['Time'] = datetime.now().strftime('%H:%M:%S')
                    dict_name['Date'] = datetime.now().strftime('%d-%m-%Y')
                    img = Image.fromarray(rgb_small_frame, 'RGB')
                    img.save(name.split('_')[0] + '.png')
                    if not any(d['name'] == name for d in name_array):
                        name_array.append(dict_name)
        process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip(face_locations, 
            face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), 
                         (right, bottom), 
                         (0, 255, 0), 3)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, 'Name: ' + name.split('_')[0], 
                        (left + 6, bottom - 6), 
                        font, .7, 
                        (255, 255, 255), 1)
            cv2.putText(frame, 'GPN: ' + name.split('_')[1], 
                        (left - 6, top - 6), 
                        font, .6, 
                        (255, 255, 255), 1)
            cv2.putText(frame, 
                        'Count of Identified Persons: ' + str(len(name_array)), 
                        (10, 30), font, .6,
                        (255, 255, 255), 1)
            cv2.putText(frame, 'Date: ' + datetime.now().strftime('%d-%m-%Y'), 
                        (10, 450), font, .6,
                        (255, 255, 255), 1)
            cv2.putText(frame, 'In Time: ' + datetime.now().strftime('%H:%M:%S'), 
                        (10, 400), font, .6,
                        (255, 255, 255), 1)
        cv2.imshow('Video', cv2.resize(frame, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end_time=time.time()
        elapsed = end_time - start_time
        if elapsed > capture_duration:
           break
    fvs.release()
    out.release()
    cv2.destroyAllWindows()
    try:
        df = pd.DataFrame(name_array)
        df = df.drop(['name'], axis=1)
        df.to_csv(path + '/Attendance.csv', index=False)
        for i in name_array:
            print(('Sending Email to {}').format(i['Name']))
            email(i['Email ID'], 
                  i['Name'], 
                  i['Time'], i['Date'])
        FileEmail()
    except:
        pass

FaceRecognition()
