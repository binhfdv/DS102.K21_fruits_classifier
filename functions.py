import glob
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

labels = []
with open('labels.txt', 'r') as lb:
    for _ in lb:
        labels.append(_.strip())
labels = sorted(labels)

def load_Data(X, Y, data_path):
    for label_i in range(len(labels)):
        for im_path in glob.glob(r"{0}\\{1}\\*.jpg".format(data_path, str(labels[label_i]))): 
            X.append(cv2.imread(im_path))
            Y.append(labels[label_i])
    return X, Y

def rbgToGrayAndResize(X):
    X_shaped = []
    for i in range(len(X)):
        grey = cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(grey,(50, 50))
        X_shaped.append(grey)
    X_shaped = np.array(X_shaped)
    return X_shaped


