from __future__ import print_function, division
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from Word2PHOC import build_phoc as PHOC
from PIL import Image, ImageOps
from torchvision.transforms import Pad

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

MAX_IMAGE_WIDTH = 1035
MAX_IMAGE_HEIGHT = 226

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
        self.h_max = 0
        self.w_max = 0

        # Get all the '.tru' files from the folder
        tru_files = glob.glob(dir_tru + "*.tru")

        for tru_file in tru_files:
            # Save the word ID
            id = os.path.splitext(os.path.basename(tru_file))[0]
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
                            # if h > self.h_max:
                            #     self.h_max = h
                            # if w > self.w_max:
                            #     self.w_max = w

    def __len__(self):
        return len(self.word_id)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dir_bmp, self.word_id[idx] + '.bmp')
        image = Image.open(img_name)
        if self.transform == 'fixed_size':
            w, h = image.size
            pad_width = MAX_IMAGE_WIDTH - w
            if pad_width < 2:
                pad_left = pad_width
                pad_right = 0
            else:
                if pad_width % 2 == 0:
                    pad_left = int(pad_width/2)
                    pad_right = pad_left
                else:
                    pad_left = int(pad_width/2) + 1
                    pad_right = pad_left - 1

            pad_height = MAX_IMAGE_HEIGHT - h
            if pad_height < 2:
                pad_top = pad_height
                pad_bottom = 0
            else:
                if pad_height % 2 == 0:
                    pad_top = int(pad_height/2)
                    pad_bottom = pad_top
                else:
                    pad_top = int(pad_height/2) + 1
                    pad_bottom = pad_top - 1

            padding = (pad_left, pad_top, pad_right, pad_bottom)
            tsfm = Pad(padding)

            # Invert the input image and then
            image = image.convert('L')
            image = ImageOps.invert(image)
            image = image.convert('1')
            image = tsfm(image)

            sample = {'image': image, 'phoc': self.phoc_word[idx], 'word': self.word_str[idx]}

        return sample

# Test the IFN/ENIT Dataset Loading

ifnenit_dataset = IfnEnitDataset(dir_tru='datasets/ifnenit_v2.0p1e/data/set_a/tru/',
                                    dir_bmp='datasets/ifnenit_v2.0p1e/data/set_a/bmp/',
                                 transform='fixed_size')

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