import tensorflow as tf
from tensorflow import keras as k


def resnet50(config, images):
    input_layer = k.layers.Input(tensor=images)
    if config.imagenet:
        resnet = k.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=input_layer)
    else:
        resnet = k.applications.resnet50.ResNet50(include_top=False, input_tensor=input_layer)
    if config.freeze:
        for layer in resnet.layers:
            layer.trainable = False
        for layer in resnet.layers[-33:]:
            layer.trainable = True

    L2 = k.regularizers.l2(config.reg)
    output = k.layers.Flatten()(resnet.output)
    output = k.layers.Dense(config.classes, name="output")(output)
    model = k.Model(inputs=resnet.input, outputs=output)

    return model


def xception(config, images):
    input_layer = k.layers.Input(tensor=images)
    if config.imagenet:
        xc = k.applications.xception.Xception(include_top=False, weights='imagenet', input_tensor=input_layer)
    else:
        xc = k.applications.xception.Xception(include_top=False, input_tensor=input_layer)
    if config.freeze:
        for layer in xc.layers:
            layer.trainable = False

    L2 = k.regularizers.l2(config.reg)
    for layer in xc.layers:
        layer.kernel_regularizer = L2
    output = k.layers.Flatten()(xc.output)
    output = k.layers.Dense(2048, activation='relu', kernel_regularizer=L2, name="fc1")(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(2048, activation='relu', kernel_regularizer=L2, name="fc2")(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(config.classes, name="output")(output)
    model = k.Model(inputs=xc.input, outputs=output)

    return model


def inceptionv2(config, images):
    input_layer = k.layers.Input(tensor=images)
    if config.imagenet:
        inceptionv2 = k.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=input_layer)
    else:
        inceptionv2 = k.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, input_tensor=input_layer)
    if config.freeze:
        for layer in inceptionv2.layers:
            layer.trainable = False

    L2 = k.regularizers.l2(config.reg)
    output = k.layers.GlobalAveragePooling2D(name='avg_pool')(inceptionv2.output)
    output = k.layers.Dense(2048, activation='relu', kernel_regularizer=L2, name="fc1")(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(2048, activation='relu', kernel_regularizer=L2, name="fc2")(output)
    output = k.layers.Dropout(config.dropout)(output)
    output = k.layers.Dense(config.classes, name="output")(output)

    model = k.Model(inputs=inceptionv2.input, outputs=output)

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
