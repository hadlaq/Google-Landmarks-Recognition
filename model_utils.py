import tensorflow as tf
from tensorflow import keras as k
from models import *
import logging


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
    elif config.model == "resnet50":
        model = resnet50(config, images)
    else:
        model = basic(config, images)

    model.compile(
        optimizer=get_optimizer(config),
        loss=get_loss,
        target_tensors=[labels],
        metrics=[get_accuracy]
    )

    return model




def save_model(model, config):
    save_path = os.path.join(config.logs_dir, get_log_path(config), 'best_model.h5')
    i = 0
    while os.path.exists(save_path):
        i += 1
        save_path = os.path.join(config.logs_dir, get_log_path(config), 'best_model' + str(i) + '.h5')

    model.save(save_path)
    logging.info('Best model saved in {}'.format(save_path))

    return save_path


def force_save_model(model, save_path):
    model.save(save_path)
    logging.info('Best model saved in {}'.format(save_path))


def load_model(config, images, labels):
    model = k.models.load_model(config.model_path, compile=False)
    model.layers.pop(0)
    new_input = k.layers.Input(tensor=images)
    new_output = model(new_input)
    model = k.Model(new_input, new_output)
    model.compile(
        optimizer=k.optimizers.SGD(0.0),  # not used
        loss=get_loss,
        target_tensors=[labels],
        metrics=[get_accuracy]
    )
    logging.info('Loaded model from {}'.format(config.model_path))
    return model


def load_model_with_no_input(config):
    model = k.models.load_model(config.model_path, compile=False)
    model.layers.pop(0)
    new_input = k.layers.Input(shape=(config.input_size, config.input_size, 3))
    new_output = model(new_input)
    model = k.Model(new_input, new_output)
    model.compile(
        optimizer=k.optimizers.SGD(0.0),  # not used
        loss=get_loss
    )
    logging.info('Loaded model from {}'.format(config.model_path))
    return model
