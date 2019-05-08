# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:24:44 2019

@author: Vishnu.Kumar1
"""

from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import os
import math

path = os.getcwd()

def DetectText(filename):
    image = cv2.imread(filename)
    orig = image.copy()
    (H, W) = image.shape[:2]
    minimum = min(H,W)
    (newW, newH) = (math.floor(minimum/32)*32, math.floor(minimum/32)*32)
    rW = W / float(newW)
    rH = H / float(newH)
    
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    
    layerNames = ["feature_fusion/Conv_7/Sigmoid",
                  "feature_fusion/concat_3"]
    net = cv2.dnn.readNet(path + "/frozen_east_text_detection.pb")
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), 
                                 swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
        
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    coordinates = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        coordinates.append((startX, startY, endX, endY))
#    cv2.imshow("Text Detection", orig)
#    cv2.waitKey(0)
    return coordinates
