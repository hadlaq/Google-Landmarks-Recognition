import tensorflow as tf


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

    return images, labels, train_init_op, dev_init_op