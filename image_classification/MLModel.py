# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:20:06 2019

@author: Vishnu.Kumar1
"""

import os
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

path = os.getcwd()
folder = path + '/DataSet/'

train = os.listdir(folder)

features = []
labels = []
for i in train:
    image = cv2.resize(cv2.imread(folder + i), 
                       (150, 150), 
                       interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    feature = hog.compute(gray)
    features.append(feature)
    if 'Approved' in i:
        labels.append(1)
    elif 'Rejected' in i:
        labels.append(0)  

features = np.array(features)
labels = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(features, labels, 
                                                    test_size=0.2, 
                                                    random_state=42)
classifiers = [KNeighborsClassifier(),
               DecisionTreeClassifier(),
               SGDClassifier(),
               LogisticRegression(multi_class='multinomial', 
                                  solver='newton-cg'),
               SVC(),
               MultinomialNB(),
               RandomForestClassifier(), 
               AdaBoostClassifier(),
               GradientBoostingClassifier(),
               BaggingClassifier(),
               MLPClassifier()]

list_class = []
for clf in classifiers:
    dict_class = {}
    name = clf.__class__.__name__
    clf.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    prediction = clf.predict(X_test.reshape(X_test.shape[0], -1))
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction, average='weighted')
    rec = recall_score(y_test, prediction, average='weighted')
    f1score = f1_score(y_test, prediction, average='weighted')
    roc_auc = roc_auc_score(y_test, prediction)
    print(name, confusion_matrix(y_test, prediction))
    dict_class['classifier'] = name
    dict_class['accuracy'] = accuracy
    dict_class['precision'] = precision
    dict_class['recall'] = rec
    dict_class['f1_score'] = f1score
    dict_class['roc_auc'] = roc_auc
    list_class.append(dict_class)
