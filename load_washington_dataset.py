from __future__ import print_function, division
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader

from Word2PHOC import build_phoc as PHOC

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

######################################################################
# Dataset class
# -------------
#
# ``torch.utils.data.Dataset`` is an abstract class representing a
# dataset.
# Your custom dataset should inherit ``Dataset`` and override the following
# methods:
#
# -  ``__len__`` so that ``len(dataset)`` returns the size of the dataset.
# -  ``__getitem__`` to support the indexing such that ``dataset[i]`` can
#    be used to get :math:`i`\ th sample
#
# Let's create a dataset class for our face landmarks dataset. We will
# read the csv in ``__init__`` but leave the reading of images to
# ``__getitem__``. This is memory efficient because all the images are not
# stored in the memory at once but read as required.
#

class WashingtonDataset(Dataset):

    def __init__(self, txt_file, root_dir, transform=None, non_alphabet=False):
        """
        Args:
            txt_file (string): Path to the text file with the GT.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.non_alphabet = non_alphabet
        self.word_id = []
        self.word_str = []
        self.phoc_word = []

        word_labels_file = open(txt_file, 'r')
        text_lines = word_labels_file.readlines()
        word_labels_file.close()

        for line in text_lines:
            # split using space to separate the ID from the letters and delete the \n
            line = line[:-1].split(" ")
            id = line[0]
            letters = line[1].split("-")

            non_alphabet_word = False
            word_string = ''
            for non_alphabet in letters:
                if "s_" in non_alphabet:
                    if "st" in non_alphabet:
                        non_alphabet = non_alphabet[2] + "st"
                    elif "nd" in non_alphabet:
                        non_alphabet = non_alphabet[2] + "nd"
                    elif "rd" in non_alphabet:
                        non_alphabet = non_alphabet[2] + "rd"
                    elif "th" in non_alphabet:
                        non_alphabet = non_alphabet[2] + "th"
                    elif non_alphabet == "s_et":
                        non_alphabet = "et"
                    elif non_alphabet == "s_s":
                        non_alphabet = 's'
                    elif non_alphabet == "s_0":
                        non_alphabet = '0'
                    elif non_alphabet == "s_1":
                        non_alphabet = '1'
                    elif non_alphabet == "s_2":
                        non_alphabet = '2'
                    elif non_alphabet == "s_3":
                        non_alphabet = '3'
                    elif non_alphabet == "s_4":
                        non_alphabet = '4'
                    elif non_alphabet == "s_5":
                        non_alphabet = '5'
                    elif non_alphabet == "s_6":
                        non_alphabet = '6'
                    elif non_alphabet == "s_7":
                        non_alphabet = '7'
                    elif non_alphabet == "s_8":
                        non_alphabet = '8'
                    elif non_alphabet == "s_9":
                        non_alphabet = '9'
                    else:
                        # If the non-alphabet flag is false I skip this image and I do not included in the dataset.
                        if self.non_alphabet:
                            if non_alphabet == "s_cm":
                                non_alphabet = ','
                            elif non_alphabet == "s_pt":
                                non_alphabet = '.'
                            elif non_alphabet == "s_sq":
                                non_alphabet = ';'
                            elif non_alphabet == "s_qo":
                                non_alphabet = ':'
                            elif non_alphabet == "s_mi":
                                non_alphabet = '-'
                            elif non_alphabet == "s_GW":
                                non_alphabet = "GW"
                            elif non_alphabet == "s_lb":
                                non_alphabet = '£'
                            elif non_alphabet == "s_bl":
                                non_alphabet = '('
                            elif non_alphabet == "s_br":
                                non_alphabet = ')'
                            elif non_alphabet == "s_qt":
                                non_alphabet = "'"
                            elif non_alphabet == "s_sl":
                                non_alphabet = "|"  # 306-03-04
                            else:
                                print(non_alphabet + "  in   " + id)
                        else:
                            non_alphabet_word = True
                            continue

                # Make sure to insert the letter in lower case
                word_string += non_alphabet.lower()

            if not non_alphabet_word:
                # Comput the PHOC of the word:
                phoc = PHOC(words=word_string)
                # print(phoc)
                # print('PHOCs has the size', np.shape(phoc))
                self.phoc_word.append(phoc)
                self.word_id.append(id)
                self.word_str.append(word_string)

    def __len__(self):
        return len(self.word_id)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.word_id[idx] + '.png')
        image = np.invert(io.imread(img_name)) / 255
        sample = {'image': image, 'phoc': self.phoc_word[idx], 'word': self.word_str[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Test the Washington Dataset Loading

washington_dataset = WashingtonDataset(txt_file='datasets/washingtondb-v1.0/ground_truth/word_labels.txt',
                                    root_dir='datasets/washingtondb-v1.0/data/word_images_normalized')

fig = plt.figure()

for i in range(len(washington_dataset)):
    sample = washington_dataset[i]

    print(i, sample['image'].shape, len(sample['word']))

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title(sample['word'])
    ax.axis('off')
    plt.imshow(sample['image'], 'gray')

    if i == 3:
        plt.show()
        break