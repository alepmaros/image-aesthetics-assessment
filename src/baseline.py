import os, cv2, random, time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.applications import vgg16, mobilenet_v2, MobileNetV2
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

ROWS = 224
COLS = 224
CHANNELS = 3

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

if __name__ == '__main__':
    
    batch_size=8

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            'datasets/photonet_flow/train', 
            target_size=(224, 224),  
            batch_size=batch_size,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            'datasets/photonet_flow/validate',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary')

    # Generate a model with all layers (with top)
    vgg16 = MobileNetV2(weights=None, include_top=True)

    #Add a layer where input is the output of the  second last layer 
    x = Dense(1, activation='sigmoid', name='predictions')(vgg16.layers[-2].output)

    #Then create the corresponding model 
    optimizer = RMSprop(lr=1e-4)
    model = Model(inputs=vgg16.input, outputs=x)
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    
    nb_epoch=10
    history = LossHistory()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    model.fit_generator(
        train_generator,
        steps_per_epoch = 10000 // batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=1000 // batch_size,
        verbose=True,
        callbacks=[history],
        class_weight = { 0: 0.273, 1: 0.727}
    )
    model.save('model1.h5')
    
    # predictions = model.predict(test_data)
    # print(predictions)

    # loss = history.losses
    # val_loss = history.val_losses
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('VGG-16 Loss Trend')
    # plt.plot(loss, 'blue', label='Training Loss')
    # plt.plot(val_loss, 'green', label='Validation Loss')
    # plt.xticks(range(0,nb_epoch)[0::2])
    # plt.legend()
    # plt.show()


