import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from custom_resnet import get_model

import config


image_input = Input(shape=(224, 224, 3))
model = get_model()

datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count,config.OUTPUT_SIZE))
    labels = np.zeros(shape=(sample_count))
    
    generator = datagen.flow_from_directory(directory,
                                            target_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
                                            batch_size=config.BATCH_SIZE,
                                            class_mode='sparse')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = model.predict(inputs_batch)
        features[i * config.BATCH_SIZE:(i + 1) * config.BATCH_SIZE] = features_batch
        labels[i * config.BATCH_SIZE:(i + 1) * config.BATCH_SIZE] = labels_batch
        i += 1
        if i * config.BATCH_SIZE >= sample_count:
            break
    
    return features, labels
