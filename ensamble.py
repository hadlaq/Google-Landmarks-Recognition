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
    parser.add_argument('--model_dir', type=str, default="./logs/model=vgg16_optimizer=adam_lr=0.001_reg=5e-05_batch_size=10_epochs=5_dropout=0.0/", help='path to model to test')
    parser.add_argument('--test_images', type=str, default="./data/test_images.csv", help='path to file of test images paths')
    parser.add_argument('--test_labels', type=str, default="./data/test_labels.csv", help='path to file of test images labels')
    parser.add_argument('--input_size', type=int, default=224, help='input is input_size x input_size x 3')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')

    return parser.parse_args()


def test(models, accs, data):
    images, labels, test_size, test_init_op = data
    k.backend.get_session().run(test_init_op)
    Y = None
    Y_pred = None
    while True:
        try:
            x, y = k.backend.get_session().run([images, labels])
            y_pred = models[0].predict(x, batch_size=x.shape[0]) * accs[0]
            for i in range(1, len(models)):
                y_pred += models[i].predict(x, batch_size=x.shape[0]) * accs[i]
            if Y is None:
                Y = y
                Y_pred = y_pred
            else:
                Y = np.concatenate((Y, y))
                Y_pred = np.concatenate((Y_pred, y_pred), axis=0)
        except tf.errors.OutOfRangeError:
            break

    logging.info(accuracy(Y_pred, Y))
    logging.info(GAP(Y_pred, Y))


def accuracy(scores, y_true):
    y_pred = np.argmax(scores, axis=1)
    rel = (y_pred == y_true).astype(int)
    return rel / y_true.shape[0]


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

    paths = [
        "",
        ""
    ]

    accs = [
        0.9,
        0.5
    ]

    s = 0
    for a in accs:
        s += a
    accs = [a / s for a in accs]
    
    models = []
    for p in paths:
        models.append(load_model_from_path(config, p))

    test(models, accs, data)


if __name__ == '__main__':
    main()
