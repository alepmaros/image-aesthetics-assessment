import os, random, time, pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard

from models.models import get_2nd_proposed_model, LossHistory
from utils.data_generators import train_generator_r, valid_generator_r, _SIZE_CV, _SIZE_TRAIN
from sys_config import _BASE_PATH

BATCH_SIZE=4

filepath = os.path.join(_BASE_PATH, 'trained_models', '2ndproposed_model.h5')

model = get_2nd_proposed_model()

nb_epoch=20

history = LossHistory()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                             mode='min', period=1)
tensorboard = TensorBoard(log_dir='Graph', histogram_freq=0, write_graph=True,
                        write_images=True)
callbacks = [history,  checkpoint]

model.fit_generator(
    train_generator_r(BATCH_SIZE),
    steps_per_epoch = _SIZE_TRAIN // BATCH_SIZE,
    epochs=nb_epoch,
    validation_data=valid_generator_r(BATCH_SIZE),
    validation_steps= _SIZE_CV // BATCH_SIZE,
    verbose=True,
    callbacks=callbacks
)

model.save(os.path.join(_BASE_PATH, 'trained_models', '2ndproposed_model_final.h5'))

loss = history.losses
val_loss = history.val_losses

with open(os.path.join('trained_models', 'loss.txt'), 'w') as fhandle:
    pickle.dump(loss, fhandle)

with open(os.path.join('trained_models', 'val_loss.txt'), 'w') as fhandle:
    pickle.dump(val_loss, fhandle)
