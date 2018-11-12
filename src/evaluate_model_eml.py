
import os, random, time, pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import keras.losses
from keras.models import load_model
from keras.models import Model

from sklearn.model_selection import train_test_split

# from utils.data_generators import test_generator_r, test_scores
from utils.DataGenerators import DataGeneratorSingleColumn

from models.losses import earth_mover_loss
keras.losses.earth_mover_loss = earth_mover_loss

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

BATCH_SIZE = 1

def get_mean(pred):
    return pred[0] * 1 + pred[1] * 2 + pred[2] * 3 + pred[4] * 5 + pred[5] * 6 + pred[6] * 7

if __name__ == '__main__':
    plt.style.use('seaborn-whitegrid')
        
    imgs_csv = pd.read_csv('datasets/photonet/photonet_cleaned_tf.csv')
    imgs_train, imgs_test = train_test_split(imgs_csv, test_size=0.2, random_state=481516)
    imgs_test, imgs_cv = train_test_split(imgs_test, test_size=0.5, random_state=481516)
    test_generator = DataGeneratorSingleColumn(imgs_test, batch_size=1)

    if os.path.exists('experiments/eml_model_predictions'):
        with open('experiments/eml_model_predictions', 'rb') as fhandle:
            y_pred = pickle.load(fhandle)
    else:
        model = load_model('trained_models/eml_model.h5')
        y_pred = model.predict_generator(test_generator, verbose=1)
        with open('experiments/eml_model_predictions', 'wb') as fhandle:
            pickle.dump(y_pred, fhandle)
    
    

    test_scores = imgs_test.apply(lambda x: x['mean_ratings'], axis=1).values
    print(len(test_scores))
    print(len(y_pred))

    residual = []
    for index, ts in enumerate(test_scores):
        mean_true = ts
        mean_predicted =  y_pred[index][0] * 1 + y_pred[index][1] * 2 + y_pred[index][2] * 3 + y_pred[index][4] * 5 + y_pred[index][5] * 6 + y_pred[index][6] * 7

        residual.append(mean_true - mean_predicted)

    plt.hist(residual, label='Média: {} $\pm$ {}'.format( np.round(np.mean(residual), 2), np.round(np.std(residual),2)))
    plt.legend()
    plt.ylabel('Quantidade')
    plt.xlabel('Resíduo')
    plt.xlim(-6, 6)
 
    with open('trained_models/loss_eml.txt', 'rb') as fhandle:
            loss = pickle.load(fhandle)
    
    with open('trained_models/val_loss_eml.txt', 'rb') as fhandle:
            val_loss = pickle.load(fhandle)

    plt.figure()
    plt.plot(range(1,21), loss, label='Loss')
    plt.plot(range(1,21), val_loss, label='Val Loss')
    plt.xticks(range(1,21))

    plt.legend()
    plt.show()