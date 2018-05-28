import tensorflow as tf
from tensorflow import keras as k


def resnet50(config, images):
    input_layer = k.layers.Input(tensor=images)
    if config.imagenet:
        resnet = k.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=input_layer)
    else:
        resnet = k.applications.resnet50.ResNet50(include_top=True, input_tensor=input_layer)
    if config.freeze:
        for layer in resnet.layers:
            layer.trainable = False
        resnet.layers[-1].trainabale = True

    L2 = k.regularizers.l2(config.reg)
    output = k.layers.Flatten()(resnet.output)
    output = k.layers.Dense(config.classes, name="output")(output)
    model = k.Model(inputs=resnet.input, outputs=output)

    return model


def vgg16(config, images):
    input_layer = k.layers.Input(tensor=images)
    if config.imagenet:
        vgg16 = k.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)
    else:
        vgg16 = k.applications.vgg16.VGG16(include_top=False, input_tensor=input_layer)

    if config.freeze:
        for layer in vgg16.layers:
            layer.trainable = False

    L2 = k.regularizers.l2(config.reg)
    output = k.layers.Flatten()(vgg16.output)
    output = k.layers.Dense(4096, activation='relu', kernel_regularizer=L2, name="fc1")(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(4096, activation='relu', kernel_regularizer=L2, name="fc2")(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(config.classes, name="output")(output)
    model = k.Model(inputs=vgg16.input, outputs=output)

    return model


def basic(config, images):
    input_layer = k.layers.Input(tensor=images, shape=(config.input_size, config.input_size, 3))

    output = k.layers.Conv2D(32, (3, 3),)(input_layer)
    output = k.layers.Activation('relu')(output)
    output = k.layers.MaxPooling2D(pool_size=(2, 2))(output)

    output = k.layers.Conv2D(32, (3, 3),)(output)
    output = k.layers.Activation('relu')(output)
    output = k.layers.MaxPooling2D(pool_size=(2, 2))(output)

    output = k.layers.Conv2D(64, (3, 3),)(output)
    output = k.layers.Activation('relu')(output)
    output = k.layers.MaxPooling2D(pool_size=(2, 2))(output)

    output = k.layers.Flatten()(output)
    output = k.layers.Dense(4096, activation='relu')(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(4096, activation='relu')(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(config.classes)(output)

    model = k.Model(inputs=input_layer, outputs=output)
    return model
