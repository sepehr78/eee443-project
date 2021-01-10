import glob
import math
import numpy as np
import os
import os.path as osp
import string
import pickle
import json
import pdb
import h5py

from torch.utils.data import DataLoader

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from matplotlib import cm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu


class Scale(object):
    """Scale transform with list as size params"""

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size[1], self.size[0]), self.interpolation)


class coco_loader(Dataset):
    """Loads train/val/test splits of coco dataset"""

    def __init__(self, data_folder, data_name, split='TRAIN', max_tokens=15):

        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open the H5 dataset file
        h5_dataset = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.ids = h5_dataset['images']

        self.ncap_per_img = h5_dataset.attrs['captions_per_image']

        # Load all captions and caption lengths into memory
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = torch.LongTensor(json.load(j))
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = torch.LongTensor(json.load(j))

        self.max_tokens = max_tokens

        self.img_transforms = transforms.Compose([
            # Scale([256, 256]),
            transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        with open(data_folder + '/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json', 'r') as j:
            self.word_dict = json.load(j)

        self.word_list = []
        for key, value in self.word_dict.items():
            temp = [key, value]
            self.word_list.append(temp)

        self.numwords = len(self.word_list)

        upd_word_list = []
        upd_word_list.append(self.word_list[-1][0])
        for i in range(0, self.numwords - 1):
            upd_word_list.append(self.word_list[i][0])
        # upd_word_list.append( self.word_list[-1][0] )
        self.word_list = upd_word_list
        # pdb.set_trace()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        img = torch.FloatTensor(self.ids[idx] / 255.0)
        img = self.img_transforms(img)
        # pdb.set_trace()
        wordclass = self.captions
        captions = []
        for add in range(self.ncap_per_img):
            single_cap = wordclass[idx + add]
            # print(single_cap)
            cap_str = ""
            for k in range(self.max_tokens):
                word_index = single_cap[k]
                if word_index != 0:
                    word = self.word_list[word_index]
                else:
                    word = self.word_list[0]
                cap_str = cap_str + " " + word
            captions.append(cap_str)

        wordclass = torch.LongTensor(self.ncap_per_img, self.max_tokens).zero_()
        sentence_mask = torch.ByteTensor(self.ncap_per_img, self.max_tokens).zero_()

        for add in range(5):
            sentence_mask[add, :(self.caplens[idx + add] + 1)] = 1

        for add in range(5):
            wordclass[add] = self.captions[idx + add, 0:self.max_tokens]
            if self.caplens[idx + add] >= self.max_tokens:
                wordclass[add, -1] = 7540  # END token

        return img, captions, wordclass, sentence_mask, idx  # 0'dı eskiden xd


if __name__ == '__main__':

    data_folder = "prep_data"
    data_name = "coco_5_cap_per_img_5_min_word_freq"
    nthreads = 0
    batchsize = 20

    train_data = coco_loader(data_folder, data_name, split='TRAIN', max_tokens=15)
    train_data_loader = DataLoader(dataset=train_data, num_workers=nthreads, batch_size=batchsize, shuffle=True,
                                   drop_last=True)

    it = iter(train_data_loader)
    first = next(it)

    references = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
    candidates = ['this', 'is', 'a', 'test']
    score = sentence_bleu(references, candidates)
    print(score)

    # TRANSFORMING INDEXED GROUND TRUTH TO SENTENCES
    test_or_val = "TEST"
    with open(os.path.join(data_folder, test_or_val + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
        true_captions = torch.LongTensor(json.load(j))

    with open(data_folder + '/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json', 'r') as j:
        word_dict = json.load(j)

    numwords = len(word_dict)
    word_list = []
    for key, value in word_dict.items():
        temp = [key, value]
        word_list.append(temp)

    upd_word_list = []
    upd_word_list.append(word_list[-1][0])
    for i in range(0, numwords - 1):
        upd_word_list.append(word_list[i][0])
    # upd_word_list.append( self.word_list[-1][0] )
    word_list = upd_word_list

    VECTOR_SIZE = 52
    END_INDEX = 7540
    CAPTION_NUM = 5
    cap_len = len(true_captions)
    txt_true_captions = []

    for i in range(int(cap_len / CAPTION_NUM)):
        caption_batch = []
        for k in range(i * CAPTION_NUM, (i + 1) * CAPTION_NUM):
            img_caption = []
            for j in range(1, VECTOR_SIZE):
                word_index = true_captions[k, j]
                if word_index != END_INDEX:
                    img_caption.append(word_list[word_index])
                else:
                    break
            caption_batch.append(img_caption)
        txt_true_captions.append(caption_batch)
