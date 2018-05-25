from __future__ import print_function, division
import os
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
        # self.h_max = 0
        # self.w_max = 0

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
            for letter in letters:
                if "s_" in letter:
                    if "st" in letter:
                        letter = letter[2] + "st"
                    elif "nd" in letter:
                        letter = letter[2] + "nd"
                    elif "rd" in letter:
                        letter = letter[2] + "rd"
                    elif "th" in letter:
                        letter = letter[2] + "th"
                    elif letter == "s_et":
                        letter = "et"
                    elif letter == "s_s":
                        letter = 's'
                    elif letter == "s_0":
                        letter = '0'
                    elif letter == "s_1":
                        letter = '1'
                    elif letter == "s_2":
                        letter = '2'
                    elif letter == "s_3":
                        letter = '3'
                    elif letter == "s_4":
                        letter = '4'
                    elif letter == "s_5":
                        letter = '5'
                    elif letter == "s_6":
                        letter = '6'
                    elif letter == "s_7":
                        letter = '7'
                    elif letter == "s_8":
                        letter = '8'
                    elif letter == "s_9":
                        letter = '9'
                    else:
                        # If the non-alphabet flag is false I skip this image and I do not included in the dataset.
                        if self.non_alphabet:
                            if letter == "s_cm":
                                letter = ','
                            elif letter == "s_pt":
                                letter = '.'
                            elif letter == "s_sq":
                                letter = ';'
                            elif letter == "s_qo":
                                letter = ':'
                            elif letter == "s_mi":
                                letter = '-'
                            elif letter == "s_GW":
                                letter = "GW"
                            elif letter == "s_lb":
                                letter = '£'
                            elif letter == "s_bl":
                                letter = '('
                            elif letter == "s_br":
                                letter = ')'
                            elif letter == "s_qt":
                                letter = "'"
                            elif letter == "s_sl":
                                letter = "|"  # 306-03-04
                            else:
                                print(letter + "  in   " + id)
                        else:
                            non_alphabet_word = True
                            continue

                # Make sure to insert the letter in lower case
                word_string += letter.lower()

            if not non_alphabet_word:
                # Compute the PHOC of the word:
                phoc = PHOC(words=word_string)
                # print(phoc)
                # print('PHOCs has the size', np.shape(phoc))
                aux_phoc_word.append(phoc)
                aux_word_id.append(id)
                aux_word_str.append(word_string)

                # Check images max size = [551, 120]
                # img_name = os.path.join(self.root_dir, id + '.png')
                # image = io.imread(img_name)
                # h, w = image.shape[:2]
                # if h > self.h_max:
                #     self.h_max = h
                # if w > self.w_max:
                #     self.w_max = w

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
        image = Image.open(img_name)
        # Invert the input image and then
        image = image.convert('L')
        image = ImageOps.invert(image)
        image = image.convert('1')

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'phoc': self.phoc_word[idx], 'word': self.word_str[idx]}

        return sample

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
    plt.figure(i);  plt.xticks([]); plt.yticks([])
    sample = washington_dataset[i]    
    plt.imshow( sample['image'].numpy()[0,:,:], 'gray')
    plt.show();
    print(i, sample['image'].shape, "; ", sample['word'], "\n")
    
    if i == 3: break






