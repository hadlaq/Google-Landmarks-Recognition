import numpy as np
import argparse
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

from model_utils import *
from data_utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, default="./logs/model=vgg16_optimizer=adam_lr=0.0001_reg=5e-05_batch_size=32_epochs=100_dropout=0.5/3/", help='path to model to test')

    parser.add_argument('--train_images', type=str, default="./data/train_images.csv", help='path to file of train images paths')
    parser.add_argument('--train_labels', type=str, default="./data/train_labels.csv", help='path to file of train images labels')
    parser.add_argument('--dev_images', type=str, default="./data/dev_images.csv", help='path to file of dev images paths')
    parser.add_argument('--dev_labels', type=str, default="./data/dev_labels.csv", help='path to file of dev images labels')
    parser.add_argument('--input_size', type=int, default=224, help='input is input_size x input_size x 3')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--max', type=int, default=100, help='max number of examples to train')

    return parser.parse_args()

def blur_image(X, sigma=1):
    X = gaussian_filter1d(X, sigma, axis=1)
    X = gaussian_filter1d(X, sigma, axis=2)
    return X


def show_image(data, label):
    images, labels, train_size, dev_size, train_init_op, dev_init_op = data
    k.backend.get_session().run(train_init_op)
    k.backend.set_learning_phase(0)
    while True:
        try:
            x, y = k.backend.get_session().run([images, labels])
            if y[0] == label:
                plt.imshow(x[0])
                plt.show()
        except tf.errors.OutOfRangeError:
            break


def saliency_map(model, data):
    images, labels, train_size, dev_size, train_init_op, dev_init_op = data
    k.backend.get_session().run(train_init_op)
    k.backend.set_learning_phase(0)
    while True:
        try:
            x, y = k.backend.get_session().run([images, labels])

            print(y[0])

            inp = model.inputs[0]
            out = model.outputs[0][:, y[0]]

            dx = k.backend.gradients(out, inp)

            smap, scores = k.backend.get_session().run([dx, model.outputs[0]], feed_dict={inp: x})

            smap = np.absolute(np.max(smap[0][0], axis=2))


            print(np.argmax(scores))
            print()

            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(x[0])
            axarr[1].imshow(x[0])
            axarr[1].imshow(smap, alpha=0.8, cmap=plt.cm.hot)
            plt.show()
        except tf.errors.OutOfRangeError:
            break


def class_viz(model, data):
    images, labels, train_size, dev_size, train_init_op, dev_init_op = data
    k.backend.get_session().run(train_init_op)
    k.backend.set_learning_phase(0)
    x, y = k.backend.get_session().run([images, labels])
    print(y[0])
    inp = model.inputs[0]
    out = model.outputs[0][:, y[0]]
    dx = k.backend.gradients(out, inp)
    plt.imshow(x[0])
    plt.show()
    noise = np.random.uniform(0, 1, size=(1, 224, 224, 3))
    blur_every = 10
    max_jitter = 16
    i = 0
    while True:
        try:
            i += 1

            ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
            noise = np.roll(np.roll(noise, ox, 1), oy, 2)

            dnoise, = k.backend.get_session().run([dx], feed_dict={inp: noise})
            noise += dnoise[0]

            noise = np.roll(np.roll(noise, -ox, 1), -oy, 2)

            if i % blur_every == 0:
                noise = blur_image(noise, sigma=0.35)

            if i % 10 == 0:
                plt.imshow(noise[0])
                plt.show()
        except tf.errors.OutOfRangeError:
            break


def confusion_matrix(model, data):
    images, labels, train_size, dev_size, train_init_op, dev_init_op = data
    k.backend.get_session().run(train_init_op)
    k.backend.set_learning_phase(0)
    mat = np.zeros((100, 100))
    i = 0
    while True:
        try:
            x, y = k.backend.get_session().run([images, labels])
            inp = model.inputs[0]
            scores, = k.backend.get_session().run([model.outputs[0]], feed_dict={inp: x})
            i += 1
            r = y[0]
            c = np.argmax(scores)
            mat[r, c] += 1
            if i == 100:
                break
        except tf.errors.OutOfRangeError:
            break
    plt.matshow(mat)
    plt.show()


def main():
    config = parse_args()

    # Getting data
    images, labels, train_size, dev_size, train_init_op, dev_init_op = get_data(config)
    data = (images, labels, train_size, dev_size, train_init_op, dev_init_op)

    # Load model
    model = load_model_with_no_input(config)

    # saliency_map(model, data)
    # confusion_matrix(model, data)
    show_image(data, 88)
    # class_viz(model, data)

if __name__ == '__main__':
    main()
