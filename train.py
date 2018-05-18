import tensorflow as tf
from tensorflow import keras as k
import argparse
import logging as log
from math import ceil
import time

from model_utils import *
from data_utils import *
from logging_utils import *

# Hide GPU warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()

    # model/training params
    parser.add_argument('--model', type=str, default="basic", help='[basic, vgg16, resnet50]')
    parser.add_argument('--optimizer', type=str, default="nesterov", help='[sgd, adam, nesterov]')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum when used')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--reg', type=float, default=5e-2, help='regularization term')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--imagenet', type=bool, default=True, help='use imagenet weights')
    parser.add_argument('--freeze', type=bool, default=True, help='freeze imagenet weights')
    parser.add_argument('--load_path', type=str, default=None, help='path to model to load (None to start from beginnig)')

    # logging
    parser.add_argument('--verbose', type=int, default=1, help='print every x batch')
    parser.add_argument('--logs_dir', type=str, default="./logs/", help='path to best model')

    # other params
    parser.add_argument('--train_images', type=str, default="./data/train_images.csv", help='path to file of train images paths')
    parser.add_argument('--train_labels', type=str, default="./data/train_labels.csv", help='path to file of train images labels')
    parser.add_argument('--dev_images', type=str, default="./data/dev_images.csv", help='path to file of dev images paths')
    parser.add_argument('--dev_labels', type=str, default="./data/dev_labels.csv", help='path to file of dev images labels')
    parser.add_argument('--input_size', type=int, default=224, help='input is input_size x input_size x 3')
    parser.add_argument('--classes', type=int, default=100, help='number of classes')

    return parser.parse_args()


def train_epoch(model, data, config, tensorboard):
    images, labels, train_size, dev_size, train_init_op, dev_init_op = data
    k.backend.get_session().run(train_init_op)
    steps = int(ceil(train_size * 1.0 / config.batch_size))
    history = model.fit(epochs=1, steps_per_epoch=steps, verbose=config.verbose,
                        callbacks=[tensorboard])

    loss = history.history['loss'][-1]
    accuracy = history.history['get_accuracy'][-1]
    return loss, accuracy


def eval_epoch(model, data, config):
    images, labels, train_size, dev_size, train_init_op, dev_init_op = data
    k.backend.get_session().run(dev_init_op)
    steps = int(ceil(dev_size * 1.0 / config.batch_size))
    loss, accuracy = model.evaluate(steps=steps, verbose=config.verbose)

    return loss, accuracy


def train(model, data, config, log_dir):
    num_epochs = config.epochs
    tensorboard = k.callbacks.TensorBoard(log_dir=log_dir, write_graph=True, write_images=False)

    train_loss_hist = []
    train_acc_hist = []
    dev_loss_hist = []
    dev_acc_hist = []
    best_dev_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs))
        tic = time.time()

        train_loss, train_acc = train_epoch(model, data, config, tensorboard)
        logging.info('Train loss %f accuracy %f' % (train_loss, train_acc))

        dev_loss, dev_acc = eval_epoch(model, data, config)
        logging.info('Dev loss %f accuracy %f' % (dev_loss, dev_acc))

        # append history
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        dev_loss_hist.append(dev_loss)
        dev_acc_hist.append(dev_acc)
        # save histories
        write_object(train_loss_hist, 'train_loss_hist', log_dir)
        write_object(train_acc_hist, 'train_acc_hist', log_dir)
        write_object(dev_loss_hist, 'dev_loss_hist', log_dir)
        write_object(dev_acc_hist, 'dev_acc_hist', log_dir)

        # save best model
        if dev_acc > best_dev_acc:
            logging.info('Best model save in epoch: {}'.format(epoch))
            best_dev_acc = dev_acc
            save_model(model, log_dir)

        logging.info("Epoch time: {}".format(time.time() - tic))


def main():
    config = parse_args()
    log_dir = set_logger(config)

    # Log full params for this run
    log.info(config)

    # Getting data
    images, labels, train_size, dev_size, train_init_op, dev_init_op = get_data(config)
    data = (images, labels, train_size, dev_size, train_init_op, dev_init_op)

    # Defining model
    model = get_model(config, images, labels)

    train(model, data, config, log_dir)


if __name__ == '__main__':
    main()
