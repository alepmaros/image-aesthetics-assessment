from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.layers import Dense, Concatenate
from keras.models import Model
from keras.applications import MobileNetV2, MobileNet
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.losses import binary_crossentropy
from keras import layers

from .losses import earth_mover_loss

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

def get_proposed_model():
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
    merged_model = Model(inputs=[model1.input, model2.input], outputs=merged_layer)

    x = Dense(7, activation='softmax')(merged_layer)
    # x = Dense(1, activation='sigmoid', name='predictions')(merged_layer)

    merged_model = Model(inputs=[model1.input, model2.input], outputs=x)

    # print(merged_model.summary())
    # from keras.utils.vis_utils import plot_model
    # plot_model(merged_model , to_file='model_plot2.png', show_shapes=True, show_layer_names=True)

    #Then create the corresponding model 
    optimizer = Adam(lr=1e-3)
    merged_model.compile(loss=earth_mover_loss,
            optimizer=optimizer)

    return merged_model


def get_2nd_proposed_model():
    model = MobileNetV2(weights=None, include_top=True)
    # x = Dense(1, activation='sigmoid', name='predictions')(model.layers[-2].output)
    x = Dense(7, activation='softmax')(model.layers[-2].output)
    final_model = Model(inputs=model.input, outputs=x)
    print(final_model.summary())
    # optimizer = Adam(lr=1e-3)
    optimizer = Adadelta()
    final_model.compile(loss=earth_mover_loss,
            optimizer=optimizer)

    return final_model


def get_baseline():
    model = MobileNetV2(weights=None, include_top=True)
    # x = Dense(1, activation='sigmoid', name='predictions')(model.layers[-2].output)
    x = Dense(7, activation='softmax')(model.layers[-2].output)
    final_model = Model(inputs=model.input, outputs=x)
    print(final_model.summary())
    optimizer = Adam(lr=1e-3)
    final_model.compile(loss=binary_crossentropy,
            optimizer=optimizer)

    return final_model

def get_model_paper():

    img_input = layers.Input(shape=(224, 224, 3))

    x = layers.Conv2D(64, (2, 2), padding='same', activation='relu')(img_input)
    x = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    # x = layers.MaxPooling2D((4, 4), strides=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(7, activation='softmax')(x)

    model = Model(img_input, x)

    optimizer = Adam(lr=1e-3)
    model.compile(loss=earth_mover_loss, optimizer=optimizer)

    return model