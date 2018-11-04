import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf

import sys 
sys.path.append('..')
# from sys_config import _BASE_PATH, _RANDOM_SEED


_BASE_PATH = '/home/apm/git/image-aesthetics-assessment'
_RANDOM_SEED = 481516

print('Loading Generators...')

IMAGE_SIZE = 224
_RANDOM_SEED_SHUFFLER = _RANDOM_SEED

def get_paths(imgs_df):
    image_paths = []
    image_scores = []
    image_scores_baseline = []
    for _, row in imgs_df.iterrows():
        img_path = os.path.join(_BASE_PATH, 'datasets/photonet/imgs/{}.jpg'.format(row['photo_id']))
        image_paths.append(img_path)

        score = np.array([
            row['nb_ratings_1'], row['nb_ratings_2'], row['nb_ratings_3'],
            row['nb_ratings_4'], row['nb_ratings_5'], row['nb_ratings_6'],
            row['nb_ratings_7']
        ])
        score = score / np.sum(score)

        image_scores.append(score.tolist())
        image_scores_baseline.append(np.rint(row['mean_ratings'])-1)
    return image_paths, image_scores, image_scores_baseline

imgs_csv = pd.read_csv(os.path.join(_BASE_PATH, 'datasets/photonet/photonet_cleaned_tf.csv'))

imgs_csv['quality'] = np.where(imgs_csv['mean_ratings'] > 5.5, 1, 0)

imgs_train, imgs_test = train_test_split(imgs_csv, test_size=0.2, random_state=_RANDOM_SEED)
imgs_test, imgs_cv = train_test_split(imgs_test, test_size=0.5, random_state=_RANDOM_SEED)

print('Train:', imgs_train.shape)
_SIZE_TRAIN = imgs_train.shape[0]
print('Test:', imgs_test.shape)
_SIZE_TEST = imgs_test.shape[0]
print('CV:', imgs_cv.shape)
_SIZE_CV = imgs_cv.shape[0]

train_image_paths, train_scores, train_scores_baseline = get_paths(imgs_train)
test_image_paths, test_scores, test_scores_baseline = get_paths(imgs_test)
cv_image_paths, cv_scores, cv_scores_baseline = get_paths(imgs_cv)

def parse_data_rc(filename, scores):
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

    image_resized = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))

    image_ccropped = tf.image.central_crop (image, 0.5)
    image_ccropped = tf.image.resize_images(image_ccropped, (IMAGE_SIZE, IMAGE_SIZE))

    return image_resized, image_ccropped, scores

def parse_data_r(filename, scores):
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
    image_resized = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))

    return image_resized, scores

def train_generator_rc(batchsize, shuffle=True):
    '''
    Args:
        batchsize: batchsize for training
        shuffle: whether to shuffle the dataset
    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        # create a dataset
        train_dataset = tf.data.Dataset().from_tensor_slices((train_image_paths, train_scores))
        train_dataset = train_dataset.map(parse_data_rc, num_parallel_calls=2)

        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=128)
        train_iterator = train_dataset.make_initializable_iterator()

        train_batch = train_iterator.get_next()

        sess.run(train_iterator.initializer)

        while True:
            try:
                X_batch_full, X_batch_cropped, y_batch = sess.run(train_batch)
                yield ([X_batch_full, X_batch_cropped], y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()

                X_batch_full, X_batch_cropped, y_batch = sess.run(train_batch)
                yield ([X_batch_full, X_batch_cropped], y_batch)

def valid_generator_rc(batchsize, shuffle=True):
    '''
    Args:
        batchsize: batchsize for training
        shuffle: whether to shuffle the dataset
    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        # create a dataset
        train_dataset = tf.data.Dataset().from_tensor_slices((cv_image_paths, cv_scores))
        train_dataset = train_dataset.map(parse_data_rc, num_parallel_calls=2)

        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=128)
        train_iterator = train_dataset.make_initializable_iterator()

        train_batch = train_iterator.get_next()

        sess.run(train_iterator.initializer)

        while True:
            try:
                X_batch_full, X_batch_cropped, y_batch = sess.run(train_batch)
                yield ([X_batch_full, X_batch_cropped], y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()

                X_batch_full, X_batch_cropped, y_batch = sess.run(train_batch)
                yield ([X_batch_full, X_batch_cropped], y_batch)

def test_generator_rc(batchsize):
    '''
    Args:
        batchsize: batchsize for training
        shuffle: whether to shuffle the dataset
    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        # create a dataset
        test_dataset = tf.data.Dataset().from_tensor_slices((test_image_paths, test_scores))
        test_dataset = test_dataset.map(parse_data_rc)

        test_dataset = test_dataset.batch(batchsize)
        test_dataset = test_dataset.repeat()

        test_iterator = test_dataset.make_initializable_iterator()

        test_batch = test_iterator.get_next()

        sess.run(test_iterator.initializer)

        while True:
            try:
                X_batch_full, X_batch_cropped, y_batch = sess.run(test_batch)
                yield ([X_batch_full, X_batch_cropped])
            except:
                test_iterator = test_dataset.make_initializable_iterator()
                sess.run(test_iterator.initializer)
                test_batch = test_iterator.get_next()

                X_batch_full, X_batch_cropped, y_batch = sess.run(test_batch)
                yield ([X_batch_full, X_batch_cropped])


def train_generator_r(batchsize, shuffle=True):
    '''
    Args:
        batchsize: batchsize for training
        shuffle: whether to shuffle the dataset
    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        # create a dataset
        train_dataset = tf.data.Dataset().from_tensor_slices((train_image_paths, train_scores))
        train_dataset = train_dataset.map(parse_data_r, num_parallel_calls=2)

        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=128)
        train_iterator = train_dataset.make_initializable_iterator()

        train_batch = train_iterator.get_next()

        sess.run(train_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()

                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)

def valid_generator_r(batchsize, shuffle=True):
    '''
    Args:
        batchsize: batchsize for training
        shuffle: whether to shuffle the dataset
    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        # create a dataset
        train_dataset = tf.data.Dataset().from_tensor_slices((cv_image_paths, cv_scores))
        train_dataset = train_dataset.map(parse_data_r, num_parallel_calls=2)

        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=128)
        train_iterator = train_dataset.make_initializable_iterator()

        train_batch = train_iterator.get_next()

        sess.run(train_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)
            except Exception as e:
                print(e)
                input()
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()

                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)


print('Generators Loaded.')

# # # Test
# from PIL import Image
# a = valid_generator_r(4)

# result = next(a)

# img_1 = (result[0][1]*255).astype('uint8')
# img_2 = (result[0][1]*255).astype('uint8')

# Image.fromarray(img_1).show()
# # Image.fromarray(img_2).show()

# # import matplotlib.pyplot as plt

# # plt.imshow(img_1)
# # plt.show()

# for i in range (0, _SIZE_CV // 4):
#     print(i, _SIZE_CV//4)
#     next(a)

