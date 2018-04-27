#!/usr/bin/python


import os.path
import zipfile
import csv
import urllib.request as urllib2
from PIL import Image
from io import BytesIO


def download_images(file_path, images_folder, list_of_categories):
    csvfile = open(file_path, 'r')
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for line in csvreader:
        image_id, image_url, landmark_id = line
        if landmark_id in list_of_categories:
            download_image(images_folder, image_id, image_url)


def download_image(images_folder, image_id, image_url):
    fname = os.path.join(images_folder, '%s.jpg' % image_id)

    if os.path.exists(fname):
        print('Image %s already exists. Skipping download.' % fname)
        return

    try:
        response = urllib2.urlopen(image_url)
        image_data = response.read()
    except:
        print('Warning: Could not download image %s' % image_url)
        return

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image %s' % image_id)
        return

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image %s to RGB' % image_id)
        return

    try:
        size = 480, 270
        pil_image_rgb.thumbnail(size, Image.ANTIALIAS)
    except:
        print('Warning: Failed to resize image %s' % image_id)
        return

    try:
        pil_image_rgb.save(fname, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image %s' % fname)
        return

    print("hi")


def get_list_of_categories(num_categories, file_path, images_folder):
    return ["10008"]


def unzip_files(data_folder):
    files = ["train.csv", "test.csv", "sample_submission.csv"]
    for f in files:
        unzip(data_folder, f)


def unzip(data_folder, fname):
    if os.path.isfile(data_folder + fname):
        print(fname + " already unzipped.")
        return

    zip_ref = zipfile.ZipFile(data_folder + fname + '.zip', 'r')
    zip_ref.extractall(data_folder)
    zip_ref.close()
    print(fname + " unzipped.")


def main():
    data_folder = '../data/metadata/'
    images_folder = '../data/images/'
    unzip_files(data_folder)

    num_categories = 5
    list_of_categories = get_list_of_categories(num_categories)

    file_path = data_folder + "train.csv"
    download_images(file_path, images_folder, list_of_categories)

if __name__ == '__main__':
    main()
