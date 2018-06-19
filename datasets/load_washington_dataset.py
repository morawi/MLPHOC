from __future__ import print_function, division

import os
import warnings

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from scripts.data_transformations import process_wg_data

warnings.filterwarnings("ignore")

class WashingtonDataset(Dataset):

    def __init__(self, cf, train=True, transform=None):
        """
        Args:
            :param cf: configuration file variables
        """

        self.root_dir = cf.dataset_path
        self.train = train  # training set or test set
        self.transform = transform
        self.non_alphabet = cf.non_alphabet
        self.word_id = []
        self.word_str = []
        self.phoc_word = []
        self.len_phoc = 0

        aux_word_id = []
        aux_word_str = []
        aux_phoc_word = []

        process_wg_data(cf, aux_phoc_word, aux_word_id, aux_word_str)

        # Use a 80% of the dataset words for testing and the other 20% for testing
        total_data = len(aux_word_id)
        np.random.seed(0)
        train_idx = np.random.choice(total_data, size=int(total_data*0.8),replace=False)
        train_idx = np.sort(train_idx)
        test_idx = []
        prev_num = -1
        for idx in range(train_idx.shape[0]):
            if idx != 0:
                prev_num = train_idx[idx-1]
            while train_idx[idx] != prev_num + 1:
                prev_num = prev_num + 1
                test_idx.append(prev_num)
        test_idx = np.sort(test_idx)

        # Choose the training or the testing indexes
        if self.train:
            data_idx = train_idx
        else:
            data_idx = test_idx

        for idx in data_idx:
            self.phoc_word.append(aux_phoc_word[idx])
            self.word_id.append(aux_word_id[idx])
            self.word_str.append(aux_word_str[idx])

        self.len_phoc = len(self.phoc_word[0])

    def num_classes(self):
        return self.len_phoc

    def __len__(self):
        return len(self.word_id)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.word_id[idx] + '.png')
        data = Image.open(img_name)
        # Invert the input image and then
        data = data.convert('L')
        data = ImageOps.invert(data)
        data = data.convert('1')

        # Convert data to numpy array
        data = np.array(data.getdata(),
                    np.uint8).reshape(data.size[1], data.size[0], 1)

        if self.transform:
            data = self.transform(data)

        # For testing give a random label
        # target = np.random.randint(1,10)

        target = self.phoc_word[idx]

        return data, target