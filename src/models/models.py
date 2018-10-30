from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.layers import Dense, Concatenate
from keras.models import Model
from keras.applications import MobileNetV2, MobileNet
from keras.optimizers import RMSprop, Adam

from models.losses import earth_mover_loss

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