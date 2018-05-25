from __future__ import print_function, division
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from torch.utils.data import Dataset, DataLoader
from Word2PHOC import build_phoc as PHOC
from PIL import Image, ImageOps
from torchvision import transforms
from data_transformations import PadImage

import globals

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class IfnEnitDataset(Dataset):

    def __init__(self, dir_tru, dir_bmp, transform=None):
        """
        Args:
            dir_tru (string): Directory with all the GT files.
            dir_bmp (string): Directory with all the BMP images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.dir_bmp = dir_bmp
        self.dir_tru = dir_tru
        self.transform = transform
        self.word_id = []
        self.word_str = []
        self.phoc_word = []
        # self.h_max = 0
        # self.w_max = 0
        # self.counter = 0

        # Get all the '.tru' files from the folder
        tru_files = glob.glob(dir_tru + "*.tru")

        for tru_file in tru_files:
            # Save the word ID
            id = os.path.splitext(os.path.basename(tru_file))[0]

            # Check if we exclude this words because is too long
            if id in globals.excluded_words_IFN_ENIT:
                continue

            self.word_id.append(id)

            # Open the tru file
            tru = open(tru_file, 'r', encoding='cp1256')
            text_lines = tru.readlines()
            tru.close()
            for line in text_lines:
                # split using space to separate the ID from the letters and delete the \n
                line = line[:-1].split(": ")
                if line[0] == "LBL":
                    tokens = line[1].split(";")
                    for token in tokens:
                        if "AW1" in str(token):
                            arabic_word = token.split(":")[1]

                            # Got an UNKNOWN UNIGRAM ERROR
                            # Compute the PHOC of the word:
                            # arabic_word = arabic_word.lower()
                            # phoc = PHOC(words=arabic_word)
                            # print(phoc)
                            # print('PHOCs has the size', np.shape(phoc))

                            phoc = ''
                            self.phoc_word.append(phoc)
                            self.word_id.append(id)
                            self.word_str.append(arabic_word)

                            # Check images max size = [1035, 226]
                            # img_name = os.path.join(self.dir_bmp, id + '.bmp')
                            # image = io.imread(img_name)
                            # h, w = image.shape[:2]
                            # if w == globals.MAX_IMAGE_WIDTH:
                            #     print("Image with max size: " + id)
                            #     self.counter = self.counter + 1
                            # if h > self.h_max:
                            #     self.h_max = h
                            # if w > self.w_max:
                            #     self.w_max = w

    def __len__(self):
        return len(self.word_id)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dir_bmp, self.word_id[idx] + '.bmp')
        image = Image.open(img_name)
        if self.transform:
            sample = self.transform(image)

        sample = {'image': image, 'phoc': self.phoc_word[idx], 'word': self.word_str[idx]}

        return sample

# Test the IFN/ENIT Dataset Loading
mean = (0.5, 0.5, 0.5)
std = (0.25, 0.25, 0.25)

pad = PadImage((globals.MAX_IMAGE_WIDTH, globals.MAX_IMAGE_HEIGHT))
composed = transforms.Compose([pad, transforms.Normalize(mean, std)])

ifnenit_dataset = IfnEnitDataset(dir_tru='datasets/ifnenit_v2.0p1e/data/set_a/tru/',
                                    dir_bmp='datasets/ifnenit_v2.0p1e/data/set_a/bmp/',
                                 transform=composed)

fig = plt.figure()

for i in range(len(ifnenit_dataset)):
    sample = ifnenit_dataset[i]

    print(i, sample['image'].size, len(sample['word']))

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title(sample['word'])
    ax.axis('off')
    plt.imshow(np.asarray((sample['image'])), 'gray')

    if i == 3:
        plt.show()
        break