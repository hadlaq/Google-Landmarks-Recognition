import tensorflow as tf
import logging


def read_images_to_lists(paths_file, labels_file, N):
    with open(paths_file) as f:
        filenames = f.readlines()
        filenames = [x.strip() for x in filenames]
    with open(labels_file) as f:
        labels = f.readlines()
        labels = [x.strip() for x in labels]
        labels_set = sorted(list(set(labels)))
        labels_dict = {}
        i = 0
        for label in labels_set:
            labels_dict[label] = i
            i += 1
        labels = [labels_dict[x] for x in labels]
    if len(filenames) > N:
        return filenames[:N], labels[:N]
    return filenames, labels


def image_parse_function(filename, label):
    image_string = tf.read_file("./data/images/" + filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # Convert to float values between 0 and 1
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return image, label


def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def get_dataset(paths_file, labels_file, batch_size, max_N, augmentation):
    filenames, labels = read_images_to_lists(paths_file, labels_file, max_N)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(image_parse_function, num_parallel_calls=4)
    if augmentation:
        dataset = dataset.map(train_preprocess, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset, len(filenames)


def get_data(config):
    dataset_train, train_size = get_dataset(config.train_images, config.train_labels, config.batch_size, config.max, config.augmentation)
    dataset_dev, dev_size = get_dataset(config.dev_images, config.dev_labels, config.batch_size, config.max, False)

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    images, labels = iterator.get_next()
    images.set_shape((None, config.input_size, config.input_size, 3))  # specify input shape

    train_init_op = iterator.make_initializer(dataset_train)
    dev_init_op = iterator.make_initializer(dataset_dev)

    logging.info("Received {} train points, {} dev points.".format(train_size, dev_size))

    return images, labels, train_size, dev_size, train_init_op, dev_init_op


def get_test_data(config):
    test_dataset, test_size = get_dataset(config.test_images, config.test_labels, config.batch_size, 900000000, False)

    iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
    images, labels = iterator.get_next()
    images.set_shape((None, config.input_size, config.input_size, 3))  # specify input shape

    test_init_op = iterator.make_initializer(test_dataset)

    logging.info("Received {} test points.".format(test_size))

    return images, labels, test_size, test_init_op
