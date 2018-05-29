from __future__ import print_function, division
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from torch.utils.data import Dataset, DataLoader
from Word2PHOC import build_phoc as PHOC
from PIL import Image, ImageOps
from torchvision import transforms
from data_transformations import PadImage, process_wg_data

import globals

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class WashingtonDataset(Dataset):

    def __init__(self, txt_file, root_dir, train=True, transform=None, non_alphabet=False):
        """
        Args:
            txt_file (string): Path to the text file with the GT.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.train = train  # training set or test set
        self.transform = transform
        self.non_alphabet = non_alphabet
        self.word_id = []
        self.word_str = []
        self.phoc_word = []
        aux_word_id = []
        aux_word_str = []
        aux_phoc_word = []

        process_wg_data(txt_file, non_alphabet, aux_phoc_word, aux_word_id, aux_word_str)

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


    def __len__(self):
        return len(self.word_id)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.word_id[idx] + '.png')
        data = Image.open(img_name)
        # Invert the input image and then
        data = data.convert('L')
        data = ImageOps.invert(data)
        data = data.convert('1')

        if self.transform:
            data = self.transform(data)

        # For testing give a random label
        target = np.random.randint(1,10)
        # target = self.phoc_word[idx]

        return data, target

# Test the Washington Dataset Loading

pad_image = PadImage((globals.MAX_IMAGE_WIDTH, globals.MAX_IMAGE_HEIGHT))

image_transfrom = transforms.Compose([pad_image,
                               transforms.ToPILImage(),
                               transforms.Scale((globals.NEW_W, globals.NEW_H)),
                               transforms.ToTensor()
])


washington_dataset = WashingtonDataset(txt_file='datasets/washingtondb-v1.0/ground_truth/word_labels.txt',
                                       root_dir='datasets/washingtondb-v1.0/data/word_images_normalized',
                                       train=True,
                                       transform=image_transfrom,
                                       non_alphabet=False)

dataloader = DataLoader(washington_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

for i in range(len(washington_dataset)):
    plt.figure(i);
    plt.xticks([]);
    plt.yticks([])
    data, target = washington_dataset[i]
    plt.imshow(data.numpy()[0, :, :], 'gray')
    plt.show();

    if i == 2: break






