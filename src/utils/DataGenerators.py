import os, cv2
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
import keras

import sys 
sys.path.append('..')
from sys_config import _BASE_PATH, _RANDOM_SEED

def parse_data(filename):
        image = tf.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize_images(image, (224, 224))

        return image

class DataGeneratorSingleColumn(keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=12, dim=(224,224), n_channels=3,
             n_classes=7, shuffle=True, seed=481516, eml=False):
        'Initialization'
        self.original_df = dataframe.copy()
        self.original_df['rint'] = self.original_df.apply(lambda x: int(np.rint(x['mean_ratings'])-1), axis=1)

        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.seed = seed
        self.eml = eml

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.fn = tf.placeholder(dtype=tf.string)
        self.tensor= parse_data(self.fn)

        self.on_epoch_end()

    def get_class_weights(self, df):
        values = df['rint']

        class_weights = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        class_weights_aux = compute_class_weight('balanced', np.unique(values), values)

        for index, unq in enumerate(np.unique(values)):
            class_weights[unq] = class_weights_aux[index]

        return class_weights

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            self.df = self.original_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            self.df = pd.concat([
                self.df[self.df['rint'] == 2],
                self.df[self.df['rint'] == 3],
                self.df[self.df['rint'] == 4],
                self.df[self.df['rint'] == 5],
                self.df[self.df['rint'] == 6],
                self.df[self.df['rint'] == 7]
            ])

            self.seed += 1
            self.class_weights = self.get_class_weights(self.df)
        else:
            self.df = self.original_df
            self.class_weights = self.get_class_weights(self.df)
    
    def __len__(self):
        return self.df.shape[0] // self.batch_size

    def __data_generation(self, idxs):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        if (self.eml):
            y = np.empty((self.batch_size, 7))
        else:
            y = np.empty((self.batch_size), dtype=int)
        weight = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i, idx in enumerate(idxs):
            # Store sample
            row = self.df.loc[idx]

            img_path = os.path.join(_BASE_PATH, 'datasets/photonet/imgs/{}.jpg'.format(row['photo_id']))
            
            img = self.sess.run(self.tensor, feed_dict={self.fn: img_path})

            X[i,] = img

            score = int(np.rint(row['mean_ratings']))-1

            if (self.eml):
                y[i] = np.array([
                        row['nb_ratings_1'], row['nb_ratings_2'], row['nb_ratings_3'],
                        row['nb_ratings_4'], row['nb_ratings_5'], row['nb_ratings_6'],
                        row['nb_ratings_7']])
                y[i] /= np.sum(y[i])
            else:
                y[i] = score

            weight[i] = self.class_weights[score]
        
        if (self.eml):
            return X, y, weight
        
        return X, keras.utils.to_categorical(y-1, num_classes=self.n_classes), weight

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = list(self.df.index.values)[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y, weight = self.__data_generation(indexes)

        return X, y, weight

    def parse_data__(self, filename, scores, weight=None):
        '''
        Loads the image file without any augmentation. Used for validation set.
        Args:
            filename: the filename from the record
            scores: the scores from the record
        Returns:
            an image referred to by the filename and its scores
        '''
        image = tf.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image_resized = tf.image.resize_images(image, (224, 224))

        return image_resized, scores, weight