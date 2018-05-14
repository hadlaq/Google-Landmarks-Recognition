import tensorflow as tf
from tensorflow import keras as k


def vgg16(config, images):
    input_layer = k.layers.Input(tensor=images, shape=(config.input_size, config.input_size, 3))

    vgg16 = k.applications.vgg16.VGG16(weights='imagenet', include_top=False,
                                       input_tensor=input_layer, input_shape=(224, 224, 3))

    for layer in vgg16.layers:
        layer.trainable = False

    output = k.layers.Flatten()(vgg16.output)
    # TODO: add regularizer?
    output = k.layers.Flatten()(output)
    output = k.layers.Dense(4096, activation='relu', kernel_regularizer=k.regularizers.l2(config.reg))(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(4096, activation='relu', kernel_regularizer=k.regularizers.l2(config.reg))(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(config.classes)(output)
    model = k.Model(inputs=vgg16.input, outputs=output)

    return model


def basic(config, images):
    input_layer = k.layers.Input(tensor=images, shape=(config.input_size, config.input_size, 3))
    output = k.layers.Flatten()(input_layer)
    output = k.layers.Dense(512)(output)
    output = k.layers.Dense(config.classes)(output)

    model = k.Model(inputs=input_layer, outputs=output)
    return model
