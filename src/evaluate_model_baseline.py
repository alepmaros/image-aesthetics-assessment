
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

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

BATCH_SIZE = 1

if __name__ == '__main__':
    plt.style.use('seaborn-whitegrid')
        
    imgs_csv = pd.read_csv('datasets/photonet/photonet_cleaned_tf.csv')
    imgs_train, imgs_test = train_test_split(imgs_csv, test_size=0.2, random_state=481516)
    imgs_test, imgs_cv = train_test_split(imgs_test, test_size=0.5, random_state=481516)
    test_generator = DataGeneratorSingleColumn(imgs_test, batch_size=1)

    if os.path.exists('experiments/baseline_model_predictions'):
        with open('experiments/baseline_model_predictions', 'rb') as fhandle:
            y_pred = pickle.load(fhandle)
    else:
        model = load_model('trained_models/baseline_model_20_0.29.h5')
        y_pred = model.predict_generator(test_generator, verbose=1)
        with open('experiments/baseline_model_predictions', 'wb') as fhandle:
            pickle.dump(y_pred, fhandle)
    
    
    test_scores = imgs_test.apply(lambda x: x['mean_ratings'], axis=1).values
    print(len(test_scores))
    print(len(y_pred))

    print(y_pred)
    
    residual = []
    for index, ts in enumerate(test_scores):
        mean_true = ts
        mean_predicted =  y_pred[index][0] * 1 + y_pred[index][1] * 2 + y_pred[index][2] * 3 + y_pred[index][4] * 5 + y_pred[index][5] * 6 + y_pred[index][6] * 7

        residual.append(mean_true - mean_predicted)

    with plt.style.context(('seaborn-darkgrid')):
        f = plt.figure(figsize=(4.5,3.5))
        plt.hist(residual,  label='Média: {} $\pm$ {}'.format( np.round(np.mean(residual), 2), np.round(np.std(residual),2)))
        plt.legend()

        plt.title('Resíduo Cross entropy')
        plt.ylabel('Quantidade')
        plt.xlabel('Resíduo')
        plt.xlim(-6, 6)
        
        plt.tight_layout
        plt.savefig('experiments/residual_baseline.pdf')