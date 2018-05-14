import tensorflow as tf
from tensorflow import keras as k
import argparse

from models import *
from data_utils import *

# Hide GPU warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')

    # training params
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')

    # logging
    parser.add_argument('--print_batch', type=bool, default=False, help='print stats after each batch')

    # other params
    parser.add_argument('--train_images', type=str, default="./data/train_images.csv", help='Path to file of train images paths')
    parser.add_argument('--train_labels', type=str, default="./data/train_labels.csv", help='Path to file of train images labels')
    parser.add_argument('--dev_images', type=str, default="./data/dev_images.csv", help='Path to file of dev images paths')
    parser.add_argument('--dev_labels', type=str, default="./data/dev_labels.csv", help='Path to file of dev images labels')
    parser.add_argument('--input_size', type=int, default=224, help='input is input_size x input_size x 3')
    parser.add_argument('--classes', type=int, default=6, help='number of classes')

    return parser.parse_args()


def train(model, data, config):
    lr = config.lr
    optimizer = tf.train.GradientDescentOptimizer(lr)
    num_epochs = config.epochs

    images, labels, train_init_op, dev_init_op = data

    # with tf.device(device):
    scores = model.output

    # Fix later to use feedable iterators
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=scores)
    loss = tf.reduce_mean(loss)
    labels_pred = tf.argmax(scores, axis=1)
    accuracy, accuracy_op = tf.metrics.accuracy(labels, labels_pred)
    train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for epoch in range(1, num_epochs + 1):
            train_epoch(sess, train_init_op, loss, train_op, accuracy, accuracy_op, config, epoch)
            eval_epoch(sess, dev_init_op, loss, accuracy, accuracy_op, config, epoch)


def train_epoch(sess, iterator_init, loss, train_op, accuracy, accuracy_op, config, epoch):
    sess.run(iterator_init)
    iterations = 0
    accuracy_sum = 0.0
    loss_sum = 0.0
    while True:
        try:
            current_loss, _, current_accuracy, _ = sess.run([loss, train_op, accuracy, accuracy_op])
            iterations += 1
            accuracy_sum += current_accuracy
            loss_sum += current_loss
            if config.print_batch:
                print('Train loss =\t', current_loss)
                print('Train running accuracy =\t', accuracy_sum / iterations)
                print('Train current accuracy =\t', current_accuracy)
        except tf.errors.OutOfRangeError:
            print('Train epoch %d loss %f accuracy %f' % (epoch, loss_sum / iterations, accuracy_sum / iterations))
            break


def eval_epoch(sess, iterator_init, loss, accuracy, accuracy_op, config, epoch):
    sess.run(iterator_init)
    iterations = 0
    accuracy_sum = 0.0
    loss_sum = 0.0
    while True:
        try:
            current_loss, current_accuracy, _ = sess.run([loss, accuracy, accuracy_op])
            iterations += 1
            accuracy_sum += current_accuracy
            loss_sum += current_loss
            if config.print_batch:
                print('Eval loss =\t', current_loss)
                print('Eval running accuracy =\t', current_loss / iterations)
                print('Eval current accuracy =\t', current_accuracy)
        except tf.errors.OutOfRangeError:
            print('Eval epoch %d loss %f accuracy %f' % (epoch, loss_sum / iterations, accuracy_sum / iterations))
            break


def main():
    config = parse_args()

    # Getting data
    images, labels, train_init_op, dev_init_op = get_data(config)
    data = (images, labels, train_init_op, dev_init_op)

    model = basic(config, images)

    train(model, data, config)

if __name__ == '__main__':
    main()
