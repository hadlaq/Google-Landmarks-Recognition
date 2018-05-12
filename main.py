import tensorflow as tf
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')

    # training params
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')

    # other params
    parser.add_argument('--train_images', type=str, default="./data/train_images.csv", help='Path to file of train images paths')
    parser.add_argument('--train_labels', type=str, default="./data/train_labels.csv", help='Path to file of train images labels')
    parser.add_argument('--dev_images', type=str, default="./data/dev_images.csv", help='Path to file of dev images paths')
    parser.add_argument('--dev_labels', type=str, default="./data/dev_labels.csv", help='Path to file of dev images labels')

    return parser.parse_args()


def read_images_to_lists(paths_file, labels_file):
    with open(paths_file) as f:
        filenames = f.readlines()
        filenames = [x.strip() for x in filenames]
    with open(labels_file) as f:
        labels = f.readlines()
        labels = [x.strip() for x in labels]
        labels_set = (labels)
        labels_dict = {}
        i = 0
        for label in labels_set:
            labels_dict[label] = i
            i += 1
        labels = [labels_dict[x] for x in labels]
    return filenames, labels


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

    iterator = dataset.make_initializable_iterator()
    return dataset, iterator

def train(model, images, labels, optimizer):
    # device = '/gpu:0'
    device = '/cpu:0'
    num_epochs = 1
    with tf.device(device):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=model)
        loss = tf.reduce_mean(loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_ops = optimizer.minimize(loss)


    with tf.session() as sess:
        sess.run(tf.global_variables_initializer())
        t = 0
        for epoch in range(num_epochs):
            print('Starting epoch %d' % epoch)
            loss_np, _ = sess.run([loss, train_ops])

def main():
    args = parse_args()
    dataset, iterator = get_dataset(args.train_images, args.train_labels, args.batch_size)

    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    # with tf.Session() as sess:
    #     sess.run(iterator_init_op)
    input_vals = tf.keras.layers.Input(tensor=images, shape=(224, 224, 3))
    print(images)
    print(input_vals)
    vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', input_tensor=input_vals, include_top=False)

    predictions = tf.keras.layers.Dense(6)(vgg16.output)
    model = tf.keras.Model(inputs=vgg16.input, outputs=predictions)
    for layer in vgg16.layers:
        layer.trainable = False

        # model.fit(steps_per_epoch=1, epochs=5, verbose=2)
    lr = 1e-3
    optimizer = tf.train.AdamOptimizer(lr)
    train(model, images, labels, optimizer)

    # input = tf.keras.Input(tensor=images)

if __name__ == '__main__':
    main()
