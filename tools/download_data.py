#!/usr/bin/python

# Edited script from Kaggle


import os.path
import zipfile


def unzip_files(data_folder):
    files = ["train.csv", "test.csv", "sample_submission.csv"]
    for f in files:
        unzip(data_folder, f)


def unzip(data_folder, fname):
    if os.path.isfile(fname):
        print(fname + " already unzipped.")
        return

    zip_ref = zipfile.ZipFile(data_folder + fname + '.zip', 'r')
    zip_ref.extractall(data_folder)
    zip_ref.close()
    print(fname + " unzipped.")


def main():
    data_folder = '../data/metadata/'
    unzip_files(data_folder)

if __name__ == '__main__':
    main()
