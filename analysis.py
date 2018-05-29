import numpy as np
import argparse
from matplotlib import pyplot as plt

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


def saliency_map(model, data):
    images, labels, train_size, dev_size, train_init_op, dev_init_op = data
    k.backend.get_session().run(dev_init_op)
    while True:
        try:
            x, y = k.backend.get_session().run([images, labels])

            inp = model.inputs[0]
            out = model.outputs[0][:, y[0]]

            dx = k.backend.gradients(out, inp)

            smap, = k.backend.get_session().run([dx], feed_dict={inp: x})
            smap = np.absolute(np.max(smap[0][0], axis=2))

            plt.imshow(x[0])
            plt.imshow(smap, alpha=0.8)
            plt.show()
        except tf.errors.OutOfRangeError:
            break


def main():
    config = parse_args()

    # Getting data
    images, labels, train_size, dev_size, train_init_op, dev_init_op = get_data(config)
    data = (images, labels, train_size, dev_size, train_init_op, dev_init_op)

    # Load model
    model = load_model_with_no_input(config)

    saliency_map(model, data)


if __name__ == '__main__':
    main()
