import tensorflow as tf
from tensorflow import keras as k
import numpy as np
import argparse
import logging as log
from math import ceil

from model_utils import *
from data_utils import *
from logging_utils import *

# Hide GPU warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()

    # logging
    parser.add_argument('--verbose', type=int, default=1, help='print every x batch')

    # other params
    parser.add_argument('--model_path', type=str, default="./logs/model=vgg16_optimizer=adam_lr=0.002_reg=5e-05_batch_size=5_epochs=2_dropout=0.0/best_model.h5", help='path to model to test')
    parser.add_argument('--model_dir', type=str, default="./logs/model=vgg16_optimizer=adam_lr=0.002_reg=5e-05_batch_size=5_epochs=2_dropout=0.0/", help='path to model to test')
    parser.add_argument('--test_images', type=str, default="./data/test_images.csv", help='path to file of test images paths')
    parser.add_argument('--test_labels', type=str, default="./data/test_labels.csv", help='path to file of test images labels')
    parser.add_argument('--input_size', type=int, default=224, help='input is input_size x input_size x 3')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')

    return parser.parse_args()


def test(model, data, config):
    images, labels, test_size, test_init_op = data
    k.backend.get_session().run(test_init_op)
    steps = int(ceil(test_size * 1.0 / config.batch_size))

    loss, accuracy = model.evaluate(steps=steps, verbose=config.verbose)
    logging.info('Test loss %f accuracy %f' % (loss, accuracy))


def test_GAP(model, data, config):
    images, labels, test_size, test_init_op = data
    k.backend.get_session().run(test_init_op)
    Y = None
    Y_pred = None
    while True:
        try:
            x = k.backend.get_session().run(images)
            y = k.backend.get_session().run(labels)
            y_pred = model.predict(x, batch_size=x.shape[0], verbose=config.verbose)
            if Y is None:
                Y = y
                Y_pred = y_pred
            else:
                Y = np.concatenate((Y, y))
                Y_pred = np.concatenate((Y_pred, y_pred), axis=0)
        except tf.errors.OutOfRangeError:
            break

    logging.info(GAP(Y_pred, Y))


def GAP(scores, y_true):
    confidence = np.max(scores, axis=1)
    y_pred = np.argmax(scores, axis=1)

    idxs = np.argsort(confidence[::-1])
    y_pred = y_pred[idxs]
    y_true = y_true[idxs]

    rel = (y_pred == y_true).astype(int)

    csum = rel.cumsum()
    M = len(y_true)
    denum = np.arange(M) + 1

    precision = csum / denum
    gap = np.sum(precision * rel) * 1.0 / M

    return gap


def main():
    config = parse_args()
    set_test_logger(config)

    # Log full params for this run
    log.info(config)

    # Getting data
    images, labels, test_size, test_init_op = get_test_data(config)
    data = (images, labels, test_size, test_init_op)

    # Load model
    # model = load_model(config, images, labels)
    # test(model, data, config)

    model = load_model_with_no_input(config)
    test_GAP(model, data, config)


if __name__ == '__main__':
    main()
