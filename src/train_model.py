import os, cv2, random, time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.layers import Dense, Concatenate
from keras.models import Model
from keras.applications import MobileNetV2, MobileNet
from keras.optimizers import RMSprop, Adam

from utils.data_generators import train_generator, valid_generator

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

ROWS = 224
COLS = 224
CHANNELS = 3
BATCH_SIZE=4

# Generate Model 1
model1 = MobileNetV2(weights=None, include_top=True)
# x = Dense(1, activation='sigmoid', name='predictions')(model1.layers[-2].output)
# model1 = Model(inputs=model1.input, outputs=x)

# Generate Model 2
model2 = MobileNetV2(weights=None, include_top=True)
for layer in model2.layers:
    layer.name = layer.name + str("_2")
# x = Dense(1, activation='sigmoid', name='predictions')(model2.layers[-2].output)
# model2 = Model(inputs=model2.input, outputs=x)

merged_layer = Concatenate()([model1.output, model2.output])
merged_model = Model([model1.input, model2.input], merged_layer)
x = Dense(1, activation='sigmoid', name='predictions')(merged_layer)

merged_model = Model(input=[model1.input, model2.input], output=x)

# print(merged_model.summary())
# from keras.utils.vis_utils import plot_model
# plot_model(merged_model , to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#Then create the corresponding model 
optimizer = Adam()
merged_model.compile(loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

nb_epoch=10
history = LossHistory()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
merged_model.fit_generator(
    train_generator(BATCH_SIZE),
    steps_per_epoch = 14362 // BATCH_SIZE,
    epochs=nb_epoch,
    validation_data=valid_generator(BATCH_SIZE),
    validation_steps=798 // BATCH_SIZE,
    verbose=True,
    callbacks=[history]
    # class_weight = { 0: 0.273, 1: 0.727}
)
merged_model.save('merged_model.h5')