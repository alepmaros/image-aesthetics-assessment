import os, cv2, random, time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard

from models.models import get_proposed_model, LossHistory
from utils.data_generators import train_generator, valid_generator, _SIZE_CV, _SIZE_TRAIN

BATCH_SIZE=4

model = get_proposed_model()

nb_epoch=20

history = LossHistory()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
checkpoint = ModelCheckpoint('proposed_model.h5', monitor='val_loss', verbose=1, save_best_only=True,
                             mode='min')
tensorboard = TensorBoard(log_dir='Graph', histogram_freq=0, write_graph=True,
                        write_images=True)
callbacks = [history,  checkpoint, tensorboard]

model.fit_generator(
    train_generator(BATCH_SIZE),
    steps_per_epoch = _SIZE_TRAIN // BATCH_SIZE,
    epochs=nb_epoch,
    validation_data=valid_generator(BATCH_SIZE),
    validation_steps= _SIZE_CV // BATCH_SIZE,
    verbose=True,
    callbacks=callbacks
)

model.save('proposed_model_final.h5')