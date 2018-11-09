
import os, random, time, pickle, cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf
import keras.losses
from keras.models import load_model
from keras.models import Model

from models.losses import earth_mover_loss

keras.losses.earth_mover_loss = earth_mover_loss

def noisy(image):
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

def get_mean(pred):
    return pred[0] * 1 + pred[1] * 2 + pred[2] * 3 + pred[4] * 5 + pred[5] * 6 + pred[6] * 7

if __name__ == '__main__':   
    model = load_model('trained_models/baseline_model.h5')

    img_good = cv2.imread('datasets/train_station.jpg')
    img_good = cv2.resize(img_good, (224, 224))
    img_good = img_good[:, :, 0:3]/255.0

    y_pred_good = model.predict(np.expand_dims(img_good, axis=0))
    print('Good', get_mean(y_pred_good[0]))

    ## Add Noise
    img_good_noisy = noisy(img_good)
    # cv2.imshow('dst_rt', img_good_noisy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    y_pred_good_noisy = model.predict(np.expand_dims(img_good_noisy, axis=0))
    print('Good noisy', get_mean(y_pred_good_noisy[0]))

    img_bad = cv2.imread('datasets/ugly_image.jpg')
    img_bad = cv2.resize(img_bad, (224, 224))
    img_bad = img_bad[:, :, 0:3]/255.0
    

    y_pred_bad = model.predict(np.expand_dims(img_bad, axis=0))
    print('Bad',  get_mean(y_pred_bad[0]))