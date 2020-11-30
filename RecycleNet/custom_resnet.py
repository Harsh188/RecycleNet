import os
import datetime
import time
from resnet50 import ResNet50
import matplotlib.pyplot as plt
import config

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Activation, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


nb_classes = 5
image_input = Input(shape=(224, 224, 3))

timestr = time.strftime("%Y-%m-%d_%H:%M:%S")

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
configuration = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def train_model():
    model = ResNet50(weights='imagenet',include_top=False)
    last_layer = model.output
    x = GlobalAveragePooling2D()(last_layer)
    x = Dense(512, activation='relu',name='fc-1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu',name='fc-2')(x)
    x = Dropout(0.5)(x)
    out = Dense(nb_classes, activation='softmax',name='output_layer')(x)
    custom_resnet_model = Model(inputs=model.input, outputs=out)

    custom_resnet_model.summary()

    #uncomment this if theres a bug
    #custom_resnet_model.layers[-1].trainable

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    train_generator=train_datagen.flow_from_directory(
                                config.TRAIN_DIR,
                                target_size=(224, 224),
                                batch_size=config.BATCH_SIZE,
                                class_mode='sparse')

    opt = tfa.optimizers.SGDW(learning_rate = 0.01,
                            weight_decay=0.0001,
                            momentum=0.9)

    # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, name='SGD')

    log_dir = config.LOGS_DIR + '/' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    custom_resnet_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=opt,
                                metrics=['accuracy'],)

    custom_resnet_model.fit(                    
        train_generator,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        epochs=config.EPOCHS,
        callbacks=[tensorboard_callback]
    )

    custom_resnet_model.save(config.TRAINED_MODEL_DIR_RESNET)

def get_model():
    model = keras.models.load_model(config.TRAINED_MODEL_DIR_RESNET)
    layer_name = 'res5c_branch2c'
    custom_resnet_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    custom_resnet_model.summary()
    return custom_resnet_model

if __name__ == '__main__':
    train_model()

