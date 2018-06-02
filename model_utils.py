import tensorflow as tf
from tensorflow import keras as k
from models import *
import logging
import os


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
    elif config.optimizer == "nesterov":
        optimizer = k.optimizers.SGD(lr, momentum=config.momentum, nesterov=True)
    else:
        optimizer = k.optimizers.SGD(lr)

    return optimizer


def get_model(config, images, labels):
    if config.model_dir is not None:
        model = load_model(config, images, labels)
        return model  # compiled
    if config.model == "vgg16":
        model = vgg16(config, images)
    elif config.model == "resnet50":
        model = resnet50(config, images)
    elif config.model == "xception":
        model = xception(config, images)
    elif config.model == "inceptionv2":
        model = inceptionv2(config, images)
    else:
        model = basic(config, images)

    model.compile(
        optimizer=get_optimizer(config),
        loss=get_loss,
        target_tensors=[labels],
        metrics=[get_accuracy]
    )

    return model


def save_model(model, log_dir):
    save_path = os.path.join(log_dir, 'best_model.h5')
    model.save(save_path)
    logging.info('Best model saved in {}'.format(save_path))

    return save_path


def load_model(config, images, labels):
    path = os.path.join(config.model_dir, 'best_model.h5')
    model = k.models.load_model(path, compile=False)
    model.layers.pop(0)
    new_input = k.layers.Input(tensor=images)
    new_output = model(new_input)
    model = k.Model(new_input, new_output)
    model.compile(
        optimizer=get_optimizer(config),
        loss=get_loss,
        target_tensors=[labels],
        metrics=[get_accuracy]
    )
    logging.info('Loaded model from {}'.format(os.path.join(config.model_dir, 'best_model.h5')))
    return model


def load_model_from_path(config, path):
    model = k.models.load_model(os.path.join(path, 'best_model.h5'), compile=False)
    model.layers.pop(0)
    new_input = k.layers.Input(shape=(config.input_size, config.input_size, 3))
    new_output = model(new_input)
    model = k.Model(new_input, new_output)
    model.compile(
        optimizer=k.optimizers.SGD(0.0),  # not used
        loss=get_loss
    )
    logging.info('Loaded model from {}'.format(os.path.join(path, 'best_model.h5')))
    return model


def load_model_with_no_input(config):
    model = k.models.load_model(os.path.join(config.model_dir, 'best_model.h5'), compile=False)
    model.layers.pop(0)
    new_input = k.layers.Input(shape=(config.input_size, config.input_size, 3))
    new_output = model(new_input)
    model = k.Model(new_input, new_output)
    model.compile(
        optimizer=k.optimizers.SGD(0.0),  # not used
        loss=get_loss
    )
    logging.info('Loaded model from {}'.format(os.path.join(config.model_dir, 'best_model.h5')))
    return model
