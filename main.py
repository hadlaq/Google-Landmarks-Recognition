import tensorflow as tf
from tensorflow import keras as k
import argparse

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


def read_images_to_lists(paths_file, labels_file):
    with open(paths_file) as f:
        filenames = f.readlines()
        filenames = [x.strip() for x in filenames]
    with open(labels_file) as f:
        labels = f.readlines()
        labels = [x.strip() for x in labels]
        labels_set = set(labels)
        labels_dict = {}
        i = 0
        for label in labels_set:
            labels_dict[label] = i
            i += 1
        labels = [labels_dict[x] for x in labels]
    return filenames[:100], labels[:100]


def image_parse_function(filename, label):
    image_string = tf.read_file("./data/images/" + filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # Convert to float values between 0 and 1
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return image, label


def get_dataset(paths_file, labels_file, batch_size):
    filenames, labels = read_images_to_lists(paths_file, labels_file)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(image_parse_function, num_parallel_calls=4)
    # dataset = dataset.map(train_preprocess, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset


def get_data(config):
    dataset_train = get_dataset(config.train_images, config.train_labels, config.batch_size)
    dataset_dev = get_dataset(config.dev_images, config.dev_labels, config.batch_size)

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    images, labels = iterator.get_next()
    images.set_shape((None, config.input_size, config.input_size, 3))  # specify input shape

    train_init_op = iterator.make_initializer(dataset_train)
    dev_init_op = iterator.make_initializer(dataset_dev)

    # iterator = dataset_train.make_initializable_iterator()
    # images, labels = iterator.get_next()
    # images.set_shape((None, config.input_size, config.input_size, 3))  # specify input shape
    # iterator_init_op = iterator.initializer

    return images, labels, train_init_op, dev_init_op


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


def vgg16(config, images):
    input_layer = k.layers.Input(tensor=images, shape=(config.input_size, config.input_size, 3))

    vgg16 = k.applications.vgg16.VGG16(weights='imagenet', include_top=False,
                                       input_tensor=input_layer, input_shape=(224, 224, 3))

    for layer in vgg16.layers:
        layer.trainable = False

    output = k.Flatten(input_shape=vgg16.output_shape[1:])(vgg16.output)
    output = k.layers.Dense(4096)(output)
    output = k.layers.Dense(4096)(output)
    output = k.layers.Dense(config.classes)(output)

    model = k.Model(inputs=vgg16.input, outputs=output)
    return model


def basic(config, images):
    input_layer = k.layers.Input(tensor=images, shape=(config.input_size, config.input_size, 3))
    output = k.layers.Flatten()(input_layer)
    output = k.layers.Dense(512)(output)
    output = k.layers.Dense(config.classes)(output)

    model = k.Model(inputs=input_layer, outputs=output)
    return model


def main():
    config = parse_args()

    # Getting data
    images, labels, train_init_op, dev_init_op = get_data(config)
    data = (images, labels, train_init_op, dev_init_op)

    model = basic(config, images)

    train(model, data, config)

if __name__ == '__main__':
    main()
