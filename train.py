import tensorflow as tf
from tensorflow import keras as k
import argparse
import logging
import pickle

from models import *
from data_utils import *
from model_utils import *

# Hide GPU warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--reg', type=float, default=5e-4, help='regularization term')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')

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
    parser.add_argument('--logs_dir', type=str, default="./logs/", help='Path to best model')

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
    labels_pred = tf.to_int32(labels_pred)
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(labels, labels_pred)))
    train_op = optimizer.minimize(loss)

    best_dev_acc = 0.0
    best_model = None

    train_loss_hist = []
    train_acc_hist = []
    dev_loss_hist = []
    dev_acc_hist = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for epoch in range(1, num_epochs + 1):
            logging.info('Epoch {}/{}'.format(epoch, num_epochs))
            train_loss, train_acc = train_epoch(sess, train_init_op, loss, train_op, accuracy, config, epoch)
            dev_loss, dev_acc = eval_epoch(sess, dev_init_op, loss, accuracy, config, epoch)

            # append history
            train_loss_hist.append(train_loss)
            train_acc_hist.append(train_acc)
            dev_loss_hist.append(dev_loss)
            dev_acc_hist.append(dev_acc)

            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                save_path = os.path.join(config.logs_dir, 'best_model.h5')
                best_model = model.save_weights(save_path) # using model.save throws errors with model.optimizer.get_config()
                logging.info('Best model saved in {}'.format(save_path))

    # Save histories
    writer(os.path.join(config.logs_dir, 'train_loss_hist'), train_loss_hist)
    writer(os.path.join(config.logs_dir, 'train_acc_hist'), train_acc_hist)
    writer(os.path.join(config.logs_dir, 'dev_loss_hist'), dev_loss_hist)
    writer(os.path.join(config.logs_dir, 'dev_acc_hist'), dev_acc_hist)

def train_epoch(sess, iterator_init, loss, train_op, accuracy, config, epoch):
    k.backend.set_learning_phase(1)
    sess.run(iterator_init)
    iterations = 0
    accuracy_sum = 0.0
    loss_sum = 0.0
    while True:
        try:
            current_loss, _, current_accuracy = sess.run([loss, train_op, accuracy])
            iterations += 1
            accuracy_sum += current_accuracy
            loss_sum += current_loss
            if config.print_batch:
                logging.info('Train loss =\t{}'.format(current_loss))
                logging.info('Train accuracy =\t{}'.format(current_accuracy))

        except tf.errors.OutOfRangeError:
            logging.info('Train epoch %d loss %f accuracy %f' % (epoch, loss_sum / iterations, accuracy_sum / iterations))
            break
    return loss_sum / iterations, accuracy_sum / iterations

def eval_epoch(sess, iterator_init, loss, accuracy, config, epoch):
    k.backend.set_learning_phase(0)
    sess.run(iterator_init)
    iterations = 0
    accuracy_sum = 0.0
    loss_sum = 0.0
    while True:
        try:
            current_loss, current_accuracy = sess.run([loss, accuracy])
            iterations += 1
            accuracy_sum += current_accuracy
            loss_sum += current_loss
            if config.print_batch:
                logging.info('Eval loss =\t{}'.format(current_loss))
                logging.info('Eval accuracy =\t{}'.format(current_accuracy))

        except tf.errors.OutOfRangeError:
            logging.info('Eval epoch %d loss %f accuracy %f' % (epoch, loss_sum / iterations, accuracy_sum / iterations))
            break
    return loss_sum / iterations, accuracy_sum / iterations


def main():
    config = parse_args()

    exp_config = 'lr='+str(config.lr)+'_reg='+str(config.reg)+'_BS='+ \
                  str(config.batch_size)+'_epochs='+str(config.epochs)+ \
                  '_drop='+str(config.dropout)
                  
    config.logs_dir = os.path.join(config.logs_dir, exp_config)
    if not os.path.exists(config.logs_dir):
        os.makedirs(config.logs_dir)
    set_logger(os.path.join(config.logs_dir, 'basic.log'))
    # Getting data
    logging.info("Creating the datasets...")
    images, labels, train_init_op, dev_init_op = get_data(config)
    data = (images, labels, train_init_op, dev_init_op)

    # Defining model
    # model = vgg16(config, images)
    model = basic(config, images)

    # Training
    logging.info("Starting training for {} epoch(s)".format(config.epochs))
    train(model, data, config)

if __name__ == '__main__':
    main()
