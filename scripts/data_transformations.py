from __future__ import print_function, division

import glob
import os

import numpy as np
from torchvision import transforms

from scripts.Word2PHOC import build_phoc as PHOC
from utils import globals

from skimage.morphology import thin as skimage_thinner


#  Method to compute the padding odf the input image to the max image size
def get_padding(image, output_size):
    output_max_width = output_size[0]
    output_max_height = output_size[1]
    h = image.shape[0]
    w = image.shape[1]

    pad_width = output_max_width - w
    if pad_width<0 : print('MAX_IMAGE_WIDTH is smaller than expected, config_file'); exit(0)
    
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
    if pad_height<0 : print('MAX_IMAGE_WIDTH is smaller than expected, see config_file'); exit(0)
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

        # tsfm = transforms.Pad(padding)        
        
        tsfm = transforms.Compose([ transforms.ToPILImage(),
                                    transforms.Pad(padding)])
        image = tsfm(image)
        image = np.array(image.getdata(),
                    np.uint8).reshape(image.size[1], image.size[0], 1)
        return image

def image_thinning(img, p):
    # thinned = skimage_thinner(image) 
    thin_iter_step  = 1   
    img=img.squeeze()
    ss = img.shape
    ss = ss[0]*ss[1]
    img_max_orig = img.max()
    for i in range(25): 
        img_max = img.max()
        sum_img = img.sum()/(img.size* img_max)
        if sum_img>p:
            img = skimage_thinner(img, max_iter= thin_iter_step)
            img = img.astype('uint8')
        else: 
            break
    
    img = img.reshape(img.shape[0], img.shape[1], 1)
    return img*img_max_orig

class ImageThinning(object):
    """ Thin the image 
        To be used as part of  torchvision.transforms
    Args: p, a threshold value to determine the thinning
        
    """
    def __init__(self, p = 0.2):
       #  assert isinstance(output_size, (int, tuple))
        self.p = p                  
        
    def __call__(self, image):
        # image =self.image_thinning(image, self.p)                      
        image = image_thinning(image, self.p)                      
        return image

def process_ifnedit_data(cf, phoc_word, word_id, word_str):
    # self.h_max = 0
    # self.w_max = 0
    # self.counter = 0

    # Get all the '.tru' files from the folder
    tru_files = glob.glob(cf.gt_path_IFN + "*.tru")

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
                        phoc = PHOC(arabic_word, cf)
                        phoc_word.append(phoc)
                        word_id.append(id)
                        word_str.append(arabic_word)

                        
def process_wg_data(cf, phoc_word, word_id, word_str):
    # self.h_max = 0
    # self.w_max = 0

    word_labels_file = open(cf.gt_path_WG, 'r')
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
                    if cf.keep_non_alphabet_in_GW:
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
            phoc = PHOC(word_string, cf)            
            phoc_word.append(phoc)
            word_id.append(id)
            word_str.append(word_string)

             
            
            
            
            
            
            