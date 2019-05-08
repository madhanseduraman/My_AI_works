# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:03:31 2019

@author: Vishnu.Kumar1
"""

from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import os
import pytesseract

path = os.getcwd()

def DetectText(filename):
    image = cv2.imread(filename)
    orig = image.copy()
    (H, W) = image.shape[:2]
    
    (newW, newH) = (768, 768)
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
    results = []
    for (startX, startY, endX, endY) in boxes:
        rtX = int(startX * rW)
        rtY = int(startY * rH)
        X = int(endX * rW)
        Y = int(endY * rH)
        dX = int((endX - startX) * 0)
        dY = int((endY - startY) * 0)
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(W, endX + (dX * 2))
        endY = min(H, endY + (dY * 2))
        roi = orig[startY:endY, startX:endX]
        config = ("-l eng --oem 1 --psm 7")
        pytesseract.pytesseract.tesseract_cmd = 'C:/Users/vishnu.kumar1/Tesseract-OCR/tesseract.exe'
        text = pytesseract.image_to_string(roi, config=config)
        results.append(((startX, startY, endX, endY), text))
    results = sorted(results, key=lambda r:r[0][1])
    for ((startX, startY, endX, endY), text) in results:
        print("OCR TEXT")
        print("========")
        print("{}\n".format(text))
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        output = orig.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(output, text, (startX, startY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow("Text Detection", output)
        cv2.waitKey(0)
        
        
#import cv2
#
#img = cv2.imread("C:/Users/vishnu.kumar1/Documents/Python Scripts/Sanitization/Images/img_1d4f5fc5100341e7ac5e8b952309c494.png", 0)
#ret, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_OTSU)
#
#
#cv2.imwrite("debug.png", thresh)