"""
Created on Thu Jul  7 12:08:22 2018

@author: malrawi

"""


from __future__ import print_function, division

import os
import warnings

import numpy as np
from PIL import Image, ImageOps, ImageChops
from torch.utils.data import Dataset
import torch
#  from scripts.Word2PHOC import build_phoc as PHOC

# from scripts.data_transformations import process_wg_data

warnings.filterwarnings("ignore")


def load_wg_data(cf):
    word_labels_file = open(cf.gt_path_WG, 'r')
    text_lines = word_labels_file.readlines()
    word_labels_file.close()
    word_id=[]; word_str=[]
    for line in text_lines:
        # split using space to separate the ID from the letters and delete the \n
        line = line[:-1].split(" ")
        id = line[0]
        letters = line[1].split("-")
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
                    if cf.keep_non_alphabet_of_GW_in_loaded_data:
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
            # Make sure to insert the letter in lower case
            word_string += letter.lower()

        word_id.append(id)
        word_str.append(word_string)
        
    return word_str, word_id


class WashingtonDataset(Dataset):

    def __init__(self, cf, train=True, transform=None, data_idx = np.arange(1), complement_idx=False):
        """
        Args:
            param cf: configuration file variables
            transform (callable, optional): Optional transform to be applied
            on a sample.
            Train(flag): generate the training set!
            data_idx: numpy.ndarray as a vector ([1, 4, 5,...])  containing the idx
            used to select the set, if none is presented, idx's will be generated 
            randomly according to split_percentage. To generate the testing set, 
            the data_idx generated by the train_set should be passed to the class 
            constructor of the test_set. 
            complement_idx: generate the set from the complement of data_idx
        """

        self.root_dir = cf.dataset_path_WG
        self.train = train  # training set or test set
        self.transform = transform
        self.keep_non_alphabet_in_GW = cf.keep_non_alphabet_of_GW_in_analysis
        self.word_id = []
        self.word_str = []
        self.weights = 1
        self.cf = cf
        
        
        aux_word_str, aux_word_id = load_wg_data(cf)                       
        len_data = len(aux_word_id)        
        
        if len(data_idx) == 1:  # this is safe as the lowest is one, when nothing is passed
            np.random.seed(cf.rnd_seed_value)
            data_idx = np.sort(np.random.choice(len_data, size=int(len_data * cf.split_percentage), 
                                                replace=False) )
            
        if complement_idx:
            all_idx = np.arange(0, len_data)
            data_idx = np.sort( np.setdiff1d(all_idx, data_idx, assume_unique=False) )

        for idx in data_idx:
            self.word_id.append(aux_word_id[idx])
            self.word_str.append(aux_word_str[idx])
        
        self.data_idx = data_idx
        self.weights = np.ones( len(data_idx), dtype = 'uint8' )
        self.PHOC_vector = torch.empty(len(data_idx), len(self.cf.PHOC('dump', self.cf)) , dtype=torch.float)
        for i in range(len(data_idx)):            
            self.PHOC_vector[i] = torch.from_numpy(self.cf.PHOC(self.word_str[i], self.cf))
            
    
        
    
    def add_weights_of_words(self): # weights to balance the loss, if the data is unbalanced   
        N = len(self.word_str)
        wordfreq = [self.word_str.count(w) for w in self.word_str]
        weights = 1 - np.array(wordfreq, dtype = 'float32')/N        
        self.weights = weights

#    def add_weights(self, weights):
#        self.weights = weights # weights to be used to balance the data, used later as input to the loss function
        
    def num_classes(self):
        if self.cf.encoder=='script_identification':
            return len(self.cf.English_label)
        else:
            return len(self.cf.PHOC('dump', self.cf)) # pasing 'dump' word to get the length
    
    def __len__(self):
        return len(self.word_id)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.word_id[index] + '.png')
        data = Image.open(img_name)
        if not(self.cf.H_gw_scale ==0): # resizing just the height            
            new_w = int(data.size[0]*self.cf.H_gw_scale/data.size[1])
            if new_w>self.cf.MAX_IMAGE_WIDTH: 
                new_w = self.cf.MAX_IMAGE_WIDTH
            data = data.resize( (new_w, self.cf.H_gw_scale), Image.ANTIALIAS)
           
        data = data.point(lambda p: p > 100 and 255) # threshold the image [0,255]
        data = data.point(lambda p: 0 if p==255 else 255 ) # invert 
        
        data = data.convert('1')      
        
        word_str = self.word_str[index]        
        if self.cf.task_type=='script_identification':
            target = self.cf.PHOC('English'.lower()+ 
                                  self.cf.language_hash_code['English'], self.cf) # language_name + hashcode
        else:
            #target = self.cf.PHOC(word_str, self.cf)
             target = self.PHOC_vector[index]
            
        if self.transform:
            data = self.transform(data)
        
        return data, target, word_str, self.weights[index]
    
    
 