import numpy as np
import pickle
import urllib.request
import h5py
import os.path
from os import path
from joblib import Parallel, delayed

from tqdm import tqdm

"""
This script downloads all the images in the training and testing datasets that is provided. It also keeps track of 
the image URLs that could not be downloaded."""

NUM_JOBS = 10  # number of processes to launch to download images
TRAIN_DATASET_PATH = "train_data"
TEST_DATASET_PATH = "test_data"
TEST_DATA_FILE_NAME = "../eee443_project_dataset_test.h5"
TRAIN_DATA_FILE_NAME = "../eee443_project_dataset.h5"


def download_img(file_path, url):
    if not path.exists(file_path):
        try:
            urllib.request.urlretrieve(url, file_path)
            return True
        except:
            return url
    return True


def get_not_downloaded_urls(train_urls, dataset_path):
    max_len = len(str(len(train_urls)))
    file_url_list = [(path.join(dataset_path, str(i).rjust(max_len, '0') + ".jpg"), url) for i, url in
                     enumerate(train_urls)]
    not_downloaded_list = []
    for file_name, url in tqdm(file_url_list):
        if not path.exists(file_name):
            not_downloaded_list.append(url)
    return not_downloaded_list


def download_images(train_urls, dataset_path):
    max_len = len(str(len(train_urls)))
    file_url_list = [(path.join(dataset_path, str(i).rjust(max_len, '0') + ".jpg"), url) for i, url in
                     enumerate(train_urls)]
    result = Parallel(n_jobs=NUM_JOBS)(delayed(download_img)(file_name, url) for file_name, url in tqdm(file_url_list))

    # filter result
    not_downloaded_urls = [url for url in result if url is not True]
    with open("not_downloaded_urls.pkl", 'wb') as output:
        pickle.dump(not_downloaded_urls, output, pickle.HIGHEST_PROTOCOL)
    # print(not_downloaded_urls)

    # for i, url in enumerate(tqdm(train_urls)):
    #     file_name = str(i).rjust(max_len, '0') + ".jpg"
    #     if not path.exists(os.path.join(dataset_path, file_name)):
    #         urllib.request.urlretrieve(url, os.path.join(dataset_path, file_name))


if __name__ == '__main__':
    with h5py.File(TRAIN_DATA_FILE_NAME, "r") as file:
        train_cap = file['train_cap'][()]
        train_imid = file['train_imid'][()]
        # train_ims = file['train_ims'][()]
        train_url = file['train_url'][()].astype(str)
        word_code = file['word_code'][()]

    with h5py.File(TEST_DATA_FILE_NAME, "r") as file:
        test_cap = file['test_caps'][()]
        test_imid = file['test_imid'][()]
        # train_ims = file['train_ims'][()]
        test_url = file['test_url'][()].astype(str)

    word_dict = {}
    for key in word_code.dtype.fields.keys():
        word_dict[key] = int(word_code[key].item())
    index_word_dict = {value: key for key, value in word_dict.items()}

    # download every image
    print("Downloading training images...")
    download_images(train_url, TRAIN_DATASET_PATH)
    not_download_url = get_not_downloaded_urls(train_url, TRAIN_DATASET_PATH)
    with open('not_available_train_urls.txt', 'w') as f:
        for item in not_download_url:
            f.write("%s\n" % item)

    print("Downloading testing images...")
    download_images(test_url, TEST_DATASET_PATH)
    not_download_url = get_not_downloaded_urls(test_url, TEST_DATASET_PATH)
    with open('not_available_test_urls.txt', 'w') as f:
        for item in not_download_url:
            f.write("%s\n" % item)

    # print(not_download_url)
    # download_images(train_url, DATASET_PATH)
