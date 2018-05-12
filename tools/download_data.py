#!/usr/bin/python

import os.path
import zipfile
import csv
import pandas as pd
import urllib.request as urllib2
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def download_images(file_path, images_folder, list_of_categories):
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
        print("Created " + images_folder)

    csvfile = open(file_path, 'r')
    csvreader = csv.reader(csvfile)
    next(csvreader)
    print_every = 100
    total = 0
    downloaded = 0
    downloaded_images = []
    for line in tqdm(csvreader):
        image_id, image_url, landmark_id = line
        if landmark_id in list_of_categories:
            total += 1
            if download_image(images_folder, image_id, image_url):
                downloaded += 1
                downloaded_images.append((image_id, landmark_id))
                if downloaded % print_every == 0:
                    write_paths_to_file(downloaded_images)
                    downloaded_images.clear()
                    print("Downloaded: ", downloaded, "/", total)
    print("Downloaded: ", downloaded, "/", total)


def write_paths_to_file(images_list):
    file_path = "../data/all.csv"
    with open(file_path, "a") as f:
        for item in images_list:
            image, label = item
            f.write(image + ".jpg" + "," + label + "\n")


def download_image(images_folder, image_id, image_url):
    fname = os.path.join(images_folder, '%s.jpg' % image_id)

    if os.path.exists(fname):
        # print('Image %s already exists. Skipping download.' % fname)
        return 0

    try:
        response = urllib2.urlopen(image_url)
        image_data = response.read()
    except:
        # print('Warning: Could not download image %s' % image_url)
        return 0

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        # print('Warning: Failed to parse image %s' % image_id)
        return 0

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        # print('Warning: Failed to convert image %s to RGB' % image_id)
        return 0

    try:
        size = 224, 224
        pil_image_small = pil_image_rgb.resize(size)
    except:
        # print('Warning: Failed to resize image %s' % image_id)
        return 0

    try:
        pil_image_small.save(fname, format='JPEG', quality=100)
    except:
        # print('Warning: Failed to save image %s' % fname)
        return 0
    return 1


def get_list_of_categories(num_categories, file_path):
    data = pd.read_csv(file_path)
    temp = data.landmark_id.value_counts().head(num_categories)
    return [str(i) for i in temp.index]


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

    num_categories = 6
    file_path = data_folder + "train.csv"
    list_of_categories = get_list_of_categories(num_categories, file_path)
    download_images(file_path, images_folder, list_of_categories)

if __name__ == '__main__':
    main()
