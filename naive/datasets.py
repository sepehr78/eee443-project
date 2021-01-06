import torch
import h5py
import json
import os
from torch.utils.data import Dataset


class CocoCaptionDataset(Dataset):
    """
    Dataset class for the COCO image captioning dataset.
    """

    def __init__(self, data_folder, data_name, split, transforms=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transforms: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open the H5 dataset file
        h5_dataset = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = h5_dataset['images']

        self.captions_per_image = h5_dataset.attrs['captions_per_image']

        # Load all captions and caption lengths into memory
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = torch.LongTensor(json.load(j))
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = torch.LongTensor(json.load(j))

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transforms = transforms

        self.dataset_size = len(self.captions)

    def __getitem__(self, index):
        # The ith caption corresponds to the (i // captions_per_image)th image
        img_idx = index // self.captions_per_image
        img = torch.FloatTensor(self.imgs[img_idx] / 255.0)

        if self.transforms is not None:
            img = self.transforms(img)

        caption = self.captions[index]
        caplen = self.caplens[index].unsqueeze(0)

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation or testing, also return all captions to find BLEU-4 score
            initial_idx = (index // self.captions_per_image) * self.captions_per_image
            all_captions = self.captions[initial_idx:initial_idx + self.captions_per_image]
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
