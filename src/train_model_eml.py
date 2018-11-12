import os, random, time, pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard

from models.models import get_2nd_proposed_model, LossHistory, get_model_paper
# from utils.data_generators import train_generator_r, valid_generator_r, _SIZE_CV, _SIZE_TRAIN, _CLASS_WEIGHTS
from utils.DataGenerators import DataGeneratorSingleColumn
from sklearn.model_selection import train_test_split

from sys_config import _BASE_PATH, _RANDOM_SEED

BATCH_SIZE=12

filepath = os.path.join(_BASE_PATH, 'trained_models', 'eml_model_{epoch:02d}_{val_loss:.2f}.h5')

model = get_model_paper()

nb_epoch=20

history = LossHistory()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                             mode='min', period=1)
tensorboard = TensorBoard(log_dir='Graph', histogram_freq=0, write_graph=True,
                        write_images=True)
callbacks = [history,  checkpoint]

imgs_csv = pd.read_csv(os.path.join(_BASE_PATH, 'datasets/photonet/photonet_cleaned_tf.csv'))

imgs_train, imgs_test = train_test_split(imgs_csv, test_size=0.15, random_state=_RANDOM_SEED)
imgs_test, imgs_cv = train_test_split(imgs_test, test_size=0.5, random_state=_RANDOM_SEED)

imgs_train['value'] = imgs_train.apply(lambda x: int(np.rint(x['mean_ratings'])), axis=1)
imgs_train = pd.concat([
    # imgs_train[imgs_train['value'] == 2].sample(frac=200.0, replace=True, random_state=481516),
    imgs_train[imgs_train['value'] == 3].sample(frac=70.0, replace=True, random_state=481516),
    imgs_train[imgs_train['value'] == 4].sample(frac=4.0, replace=True, random_state=481516),
    imgs_train[imgs_train['value'] == 5],
    imgs_train[imgs_train['value'] == 6].sample(frac=2.0, replace=True, random_state=481516),
    imgs_train[imgs_train['value'] == 7].sample(frac=30.0, replace=True, random_state=481516),
])
imgs_train.drop(columns=['value'], inplace=True)

train_generator = DataGeneratorSingleColumn(imgs_train, eml=True)
valid_generator = DataGeneratorSingleColumn(imgs_cv, eml=True)

model.fit_generator(
    train_generator,
    validation_data = valid_generator,
    epochs=nb_epoch,
    verbose=True,
    callbacks=callbacks
)

model.save(os.path.join(_BASE_PATH, 'trained_models', 'eml_model_final.h5'))

loss = history.losses
val_loss = history.val_losses

with open(os.path.join('trained_models', 'loss_eml.txt'), 'wb') as fhandle:
    pickle.dump(loss, fhandle)

with open(os.path.join('trained_models', 'val_loss_eml.txt'), 'wb') as fhandle:
    pickle.dump(val_loss, fhandle)