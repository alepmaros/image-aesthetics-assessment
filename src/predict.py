
import os, random, time, pickle, cv2, copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import PIL

import tensorflow as tf
import keras.losses
from keras.models import load_model
from keras.models import Model

from models.losses import earth_mover_loss
from utils.DataGenerators import DataGeneratorSingleColumn
from sklearn.model_selection import train_test_split

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

def parse_data(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize_images(image, (224, 224))

    return image

if __name__ == '__main__':   
    

    with tf.Session() as sess:
        

        sess.run(tf.global_variables_initializer())
        fn = tf.placeholder(dtype=tf.string)
        tensor = parse_data(fn)

        # imgs_csv = pd.read_csv('datasets/photonet/photonet_cleaned_tf.csv')
        # imgs_train, imgs_test = train_test_split(imgs_csv, test_size=0.2, random_state=481516)
        # imgs_test, imgs_cv = train_test_split(imgs_test, test_size=0.5, random_state=481516)

        # # imgs_cv.sort_values('mean_ratings', ascending=False, inplace=True)

        # imgs_cv = pd.concat([
        #     imgs_cv.sort_values('mean_ratings', ascending=False).head(5),
        #     imgs_cv.sort_values('mean_ratings', ascending=True).head(5)
        # ])

        # cv_generator = DataGeneratorSingleColumn(imgs_cv.head(10), batch_size=1)

        

        # y_pred = model.predict_generator(cv_generator, verbose=1)

        # for yy in y_pred:
        #     print(get_mean(yy))

        model = load_model('trained_models/eml_model_20_0.13.h5')
        
        X = np.empty((4, 224, 224, 3))
        X[0,] = sess.run(tensor, feed_dict={fn: 'datasets/cat_example.jpg'})
        X[1,] = sess.run(tensor, feed_dict={fn: 'datasets/cat_example_blurred.jpg'})
        X[2,] = sess.run(tensor, feed_dict={fn: 'datasets/cat_example_exposure.jpg'})
        X[3,] = sess.run(tensor, feed_dict={fn: 'datasets/ugly_image.jpg'})

        # img = PIL.Image.fromarray(np.uint8(X[0,]*255))
        # img.show()
        # img = PIL.Image.fromarray(np.uint8(X[1,]*255))
        # img.show()

        y_pred = model.predict_on_batch(X)
        print(y_pred)
        print('Cat Example', get_mean(y_pred[0]), y_pred[0])
        print('Cat Example Blurred', get_mean(y_pred[1]), y_pred[1])
        print('Cat Example Noise', get_mean(y_pred[2]), y_pred[2])
        print('Ugly', get_mean(y_pred[3]), y_pred[3])