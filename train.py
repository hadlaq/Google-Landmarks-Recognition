import tensorflow as tf
from tensorflow import keras as k
import argparse
import logging as log

from model_utils import *
from data_utils import *
from logging_utils import *

# Hide GPU warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()

    # model/training params
    parser.add_argument('--model', type=str, default="vgg16", help='[basic, vgg16]')
    parser.add_argument('--optimizer', type=str, default="adam", help='[sgd, adam]')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--reg', type=float, default=5e-5, help='regularization term')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--imagenet', type=bool, default=True, help='Use imagenet weights')
    parser.add_argument('--freeze', type=bool, default=True, help='Freeze imagenet weights')

    # logging
    parser.add_argument('--verbose', type=int, default=1, help='print every x batch')
    parser.add_argument('--logs_dir', type=str, default="./logs/", help='path to best model')

    # other params
    parser.add_argument('--train_images', type=str, default="./data/train_images.csv", help='path to file of train images paths')
    parser.add_argument('--train_labels', type=str, default="./data/train_labels.csv", help='path to file of train images labels')
    parser.add_argument('--dev_images', type=str, default="./data/dev_images.csv", help='path to file of dev images paths')
    parser.add_argument('--dev_labels', type=str, default="./data/dev_labels.csv", help='path to file of dev images labels')
    parser.add_argument('--input_size', type=int, default=224, help='input is input_size x input_size x 3')
    parser.add_argument('--classes', type=int, default=6, help='number of classes')

    return parser.parse_args()


def train_epoch(model, data, config):
    images, labels, train_size, dev_size, train_init_op, dev_init_op = data
    k.backend.get_session().run(train_init_op)
    steps = int(train_size * 1.0 / config.batch_size)
    history = model.fit(epochs=1, steps_per_epoch=steps, verbose=config.verbose)

    loss = history.history['loss'][-1]
    accuracy = history.history['get_accuracy'][-1]
    return loss, accuracy


def eval_epoch(model, data, config):
    images, labels, train_size, dev_size, train_init_op, dev_init_op = data
    k.backend.get_session().run(dev_init_op)
    steps = int(dev_size * 1.0 / config.batch_size)
    loss, accuracy = model.evaluate(steps=steps, verbose=config.verbose)

    return loss, accuracy


def train(model, data, config):
    num_epochs = config.epochs

    train_loss_hist = []
    train_acc_hist = []
    dev_loss_hist = []
    dev_acc_hist = []
    best_dev_acc = 0.0
    best_model = None
    for epoch in range(1, num_epochs + 1):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs))

        train_loss, train_acc = train_epoch(model, data, config)
        logging.info('Train loss %f accuracy %f' % (train_loss, train_acc))

        dev_loss, dev_acc = eval_epoch(model, data, config)
        logging.info('Dev loss %f accuracy %f' % (dev_loss, dev_acc))

        # append history
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        dev_loss_hist.append(dev_loss)
        dev_acc_hist.append(dev_acc)

        # save best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            if best_model is None:
                best_model = save_model(model, config)
            else:
                force_save_model(model, best_model)

    # save histories
    write_object(train_loss_hist, 'train_loss_hist', config)
    write_object(train_acc_hist, 'train_acc_hist', config)
    write_object(dev_loss_hist, 'dev_loss_hist', config)
    write_object(dev_acc_hist, 'dev_acc_hist', config)


def main():
    config = parse_args()
    set_logger(config)

    # Log full params for this run
    log.info(config)

    # Getting data
    images, labels, train_size, dev_size, train_init_op, dev_init_op = get_data(config)
    data = (images, labels, train_size, dev_size, train_init_op, dev_init_op)

    # Defining model
    model = get_model(config, images, labels)

    train(model, data, config)


if __name__ == '__main__':
    main()
