import tensorflow as tf
from tensorflow import keras as k


def vgg16(config, images):
    input_layer = k.layers.Input(tensor=images, shape=(config.input_size, config.input_size, 3))

    vgg16 = k.applications.vgg16.VGG16(weights='imagenet', include_top=False,
                                       input_tensor=input_layer, input_shape=(224, 224, 3))

    for layer in vgg16.layers:
        layer.trainable = True

    output = k.layers.Flatten()(vgg16.output)
    # TODO: add regularizer?
    output = k.layers.Dense(4096, activation='relu', kernel_regularizer=k.regularizers.l2(config.reg))(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(4096, activation='relu', kernel_regularizer=k.regularizers.l2(config.reg))(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(config.classes)(output)
    model = k.Model(inputs=vgg16.input, outputs=output)

    return model

def vgg16_m(config, images):
    input_layer = k.layers.Input(tensor=images, shape=(config.input_size, config.input_size, 3))

    vgg16 = k.applications.vgg16.VGG16(weights='imagenet', include_top=False,
                                       input_tensor=input_layer, input_shape=(224, 224, 3))

    for layer in vgg16.layers:
        layer.trainable = True

    vgg16.layers.pop()
    vgg16.layers.pop()
    vgg16.layers.pop()
    vgg16.layers.pop()

    output = k.layers.Conv2D(512, (3, 3), activation='relu', padding='same',)(vgg16.get_layer('block4_pool').output)
    # output = k.layers.Conv2D(512, (3, 3), activation='relu', padding='same',)(vgg16.output)
    output = k.layers.Conv2D(512, (3, 3), activation='relu', padding='same',)(output)
    output = k.layers.Conv2D(512, (3, 3), activation='relu', padding='same',)(output)
    output = k.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(output)

    output = k.layers.Flatten()(output)
    # TODO: add regularizer?
    output = k.layers.Dense(4096, activation='relu', kernel_regularizer=k.regularizers.l2(config.reg))(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(4096, activation='relu', kernel_regularizer=k.regularizers.l2(config.reg))(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(config.classes)(output)
    model = k.Model(inputs=vgg16.input, outputs=output)

    return model

def vgg18(config, images):
    input_layer = k.layers.Input(tensor=images, shape=(config.input_size, config.input_size, 3))

    vgg16 = k.applications.vgg16.VGG16(weights='imagenet', include_top=False,
                                       input_tensor=input_layer, input_shape=(224, 224, 3))

    for layer in vgg16.layers:
        layer.trainable = False

    output = k.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=k.regularizers.l2(config.reg))(vgg16.output)
    output = k.layers.MaxPooling2D()(output)
    output = k.layers.Flatten()(output)
    # TODO: add regularizer?
    output = k.layers.Dense(4096, activation='relu', kernel_regularizer=k.regularizers.l2(config.reg))(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(4096, activation='relu', kernel_regularizer=k.regularizers.l2(config.reg))(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(config.classes)(output)
    model = k.Model(inputs=vgg16.input, outputs=output)

    return model

def resnet50(config, images):
    input_layer = k.layers.Input(tensor=images, shape=(config.input_size, config.input_size, 3))
    resnet = k.applications.resnet50.ResNet50(include_top=False, weights='imagenet',
                                                input_tensor=input_layer, input_shape=(224, 224, 3))

    for layer in resnet.layers:
        layer.trainable = False

    output = k.layers.Flatten()(resnet.output)
    # TODO: add regularizer?
    output = k.layers.Dense(4096, activation='relu', kernel_regularizer=k.regularizers.l2(config.reg))(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(4096, activation='relu', kernel_regularizer=k.regularizers.l2(config.reg))(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(config.classes)(output)
    model = k.Model(inputs=resnet.input, outputs=output)

    return model

def basic(config, images):
    input_layer = k.layers.Input(tensor=images, shape=(config.input_size, config.input_size, 3))
    output = k.layers.Flatten()(input_layer)
    output = k.layers.Dense(512)(output)
    output = k.layers.Dense(config.classes)(output)

    model = k.Model(inputs=input_layer, outputs=output)
    return model
