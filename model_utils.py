import tensorflow as tf
from tensorflow import keras as k
from models import *


def get_loss(y_true, y_pred):
    return tf.keras.backend.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


def get_accuracy(y_true, y_pred):
    y_pred = tf.to_int32(tf.argmax(y_pred, axis=1))
    accuracy = k.backend.mean(tf.to_float(tf.equal(y_true, y_pred)))
    return accuracy


def get_optimizer(config):
    lr = config.lr
    if config.optimizer == 'adam':
        optimizer = k.optimizers.Adam(lr)
    else:
        optimizer = k.optimizers.SGD(lr)

    return optimizer


def get_model(config, images, labels):
    if config.model == "vgg16":
        model = vgg16(config, images)
    else:
        model = basic(config, images)

    model.compile(
        optimizer=get_optimizer(config),
        loss=get_loss,
        target_tensors=[labels],
        metrics=[get_accuracy]
    )

    return model
