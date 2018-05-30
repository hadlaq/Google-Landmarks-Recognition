#!/usr/bin/python


import pandas as pd
import random


def writer(path, list):
    my_df = pd.DataFrame(list)
    my_df.to_csv(path, index=False, header=False)


def shuff(list1, list2):
    zipped = list(zip(list1, list2))
    random.shuffle(zipped)
    tup1, tup2 = zip(*zipped)
    return list(tup1), list(tup2)


def main():
    data = pd.read_csv('../data/all.csv', header=None, usecols=[0, 1])

    images = list(data[0])
    labels = list(data[1])
    images, labels = shuff(images, labels)

    train_split = 0.9
    dev_split = 0.02

    train_limit = int(train_split * len(images))
    dev_limit = train_limit + int(dev_split * len(images))

    train_images = images[:train_limit]
    train_labels = labels[:train_limit]
    print('size of train set: ', len(train_images))
    
    dev_images = images[train_limit:dev_limit]
    dev_labels = labels[train_limit:dev_limit]
    print('size of dev set: ', len(dev_images))

    test_images = images[dev_limit:]
    test_labels = labels[dev_limit:]
    print('size of test set: ', len(test_images))

    writer('../data/train_images.csv', train_images)
    writer('../data/train_labels.csv', train_labels)

    writer('../data/dev_images.csv', dev_images)
    writer('../data/dev_labels.csv', dev_labels)

    writer('../data/test_images.csv', test_images)
    writer('../data/test_labels.csv', test_labels)


def test_shuff():
    a = [1,2,3]
    b = [4,5,6]

    a, b = shuff(a, b)
    print(a, b)


if __name__ == '__main__':
    main()
    # test_shuff()
