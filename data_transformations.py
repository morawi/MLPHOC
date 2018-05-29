from __future__ import print_function, division
from torchvision import transforms
from Word2PHOC import build_phoc as PHOC
import numpy as np
import os
import glob
import globals

#  Method to compute the padding odf the input image to the max image size
def get_padding(image, output_size):
    output_max_width = output_size[0]
    output_max_height = output_size[1]
    w, h = image.size
    pad_width = output_max_width - w
    if pad_width < 2:
        pad_left = pad_width
        pad_right = 0
    else:
        if pad_width % 2 == 0:
            pad_left = int(pad_width / 2)
            pad_right = pad_left
        else:
            pad_left = int(pad_width / 2) + 1
            pad_right = pad_left - 1

    pad_height = output_max_height - h
    if pad_height < 2:
        pad_top = pad_height
        pad_bottom = 0
    else:
        if pad_height % 2 == 0:
            pad_top = int(pad_height / 2)
            pad_bottom = pad_top
        else:
            pad_top = int(pad_height / 2) + 1
            pad_bottom = pad_top - 1

    padding = (pad_left, pad_top, pad_right, pad_bottom)
    return padding

# Class to perform the padding
class PadImage(object):
    """Pad the image in a sample to the max size

    Args:
        output_size (tuple or int): Desired output size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_max_width = output_size[0]
        self.output_max_height = output_size[1]

    def __call__(self, image):
        padding = get_padding(image, (self.output_max_width, self.output_max_height))
        tsfm = transforms.Pad(padding)
        image = tsfm(image)
        image = np.array(image.getdata(),
                    np.uint8).reshape(image.size[1], image.size[0], 1)
        return image

def process_ifnedit_data(dir_tru, phoc_word, word_id, word_str):
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
                        phoc_word.append(phoc)
                        word_id.append(id)
                        word_str.append(arabic_word)

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


def process_wg_data(txt_file, non_alphabet, phoc_word, word_id, word_str):

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
                    if non_alphabet:
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
            phoc_word.append(phoc)
            word_id.append(id)
            word_str.append(word_string)

            # Check images max size = [551, 120]
            # img_name = os.path.join(self.root_dir, id + '.png')
            # image = io.imread(img_name)
            # h, w = image.shape[:2]
            # if h > self.h_max:
            #     self.h_max = h
            # if w > self.w_max:
            #     self.w_max = w