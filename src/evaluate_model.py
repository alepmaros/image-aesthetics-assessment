import os, cv2, random, time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.models import load_model
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

ROWS = 224
COLS = 224
CHANNELS = 3

if __name__ == '__main__':
    plt.style.use('seaborn-whitegrid')
    model = load_model('model1.h5')

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
            'datasets/photonet_flow/test', 
            target_size=(224, 224),  
            class_mode='binary',
            shuffle=False)


    y_test = test_generator.classes
    y_pred = model.predict_generator(test_generator, verbose=1)
    
    ## ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Area = {}'.format(roc_auc))
    plt.legend()
    plt.show()

    ## Precision Recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
    plt.show()