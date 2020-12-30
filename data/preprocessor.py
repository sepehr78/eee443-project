import os
import numpy as np
import h5py
import json
import torch
from tqdm import tqdm
from collections import Counter
# from random import seed, choice, sample
from PIL import Image
import torchvision.transforms as transforms


def create_input_files(dataset, karpathy_json_path, image_folder, output_folder,
                       captions_per_image=5, min_word_freq=5, max_len=50, seed=34343):
    """
    Preprocesses the images and captions as follows: Each image is resized to 256 by 256 and saved as a part of an h5
    dataset. Each caption is tokenized and punctuations/spaces removed. Then, each word in the caption that appears
    more than a given threshold (default=5) in the caption is mapped to an integer. All captions are padded to a
    given length (default=50) and start/end markers are added. Captions are saved as .json files.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []  # contains train & val image paths, to be split later
    train_image_captions = []
    test_image_paths = []
    test_image_captions = []
    train_word_freq = Counter()
    for img in tqdm(data['images']):
        captions = []
        for c in img['sentences']:
            # Update word frequency
            if 'train' in img['filename']:
                train_word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if 'train' in img['filename']:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        else:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in train_word_freq.keys() if train_word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<pad>'] = 0
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    rng = np.random.default_rng(seed)
    resize_trans = transforms.Compose([
        transforms.Resize((256, 256))
    ])
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + rng.choice(imcaps[i], captions_per_image - len(imcaps[i]),
                                                      replace=False).tolist()
                else:
                    captions = rng.choice(imcaps[i], captions_per_image, replace=False)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = Image.open(impaths[i]).convert("RGB")
                trans_img = resize_trans(img)
                np_img = np.array(trans_img).transpose(2, 0, 1)
                assert len(np_img.shape) == 3

                # Save image to HDF5 file
                images[i] = np_img

                for c in captions:
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


if __name__ == '__main__':
    # assumes the coco dataset is saved in /data/coco2014 relative to the working directory will save the outputs
    # in a folder called saved_output
    # create_input_files("coco", "data/karpathy/dataset_coco.json", "data/coco2014/", "saved_output/")
    # print("donee")

    # read the training dataset and open a random image
    train_h5 = h5py.File("show_tell/saved_output/TRAIN_IMAGES_coco_5_cap_per_img_5_min_word_freq.hdf5", 'r')
    imgs = train_h5['images']
    rand_idx = np.random.choice(range(len(imgs)))
    rand_img = imgs[rand_idx]
    # transpose is needed to go from (C, H, W) to (H, W, C)
    rand_img = Image.fromarray(rand_img.transpose(1, 2, 0))
    rand_img.show()
