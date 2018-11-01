import os, random, time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import keras.losses
from keras.models import load_model
from keras.models import Model

from utils.data_generators import test_generator_rc, test_scores
from models.losses import earth_mover_loss

keras.losses.earth_mover_loss = earth_mover_loss

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

BATCH_SIZE = 4

if __name__ == '__main__':
    plt.style.use('seaborn-whitegrid')
    model = load_model('trained_models/proposed_model.h5')
    
    y_pred = model.predict_generator(test_generator_rc(BATCH_SIZE), steps=(len(test_scores)//BATCH_SIZE)-2, verbose=1)
    
    print(len(test_scores))
    print(len(y_pred))

    print(y_pred)

    for index, ts in enumerate(test_scores):
        mean_true = ts[0] * 1 + ts[1] * 2 + ts[2] * 3 + ts[4] * 5 + ts[5] * 6 + ts[6] * 7
        mean_predicted =  y_pred[index][0] * 1 + y_pred[index][1] * 2 + y_pred[index][2] * 3 + y_pred[index][4] * 5 + y_pred[index][5] * 6 + y_pred[index][6] * 7

        print(mean_true, mean_predicted)
        input()