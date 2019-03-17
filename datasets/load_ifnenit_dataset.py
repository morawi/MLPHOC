"""
Created on Thu Jul  7 12:08:22 2018

@author: malrawi

"""


from __future__ import print_function, division

import os
import warnings
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils import globals
import torchvision.transforms as transforms

# from scripts.data_transformations import process_ifnedit_data

warnings.filterwarnings("ignore")

def load_ifnedit_data(cf):
    word_id = []
    word_str = []
    phoc_word = []
   
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
                        # phoc = cf.PHOC(arabic_word, cf)
                        # phoc_word.append(phoc)
                        word_id.append(id)
                        word_str.append(arabic_word)
    return phoc_word, word_id, word_str

class IfnEnitDataset(Dataset):

    def __init__(self, cf, train=True, transform=None, data_idx= np.arange(1), complement_idx=False):
        """
        Args:
            dir_tru (string): Directory with all the GT files.
            dir_bmp (string): Directory with all the BMP images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
            data_idx: numpy.ndarray as a vector ([1, 4, 5,...])  containing the idx
            used to select the set, if none is presented, idx's will be generated 
            randomly according to split_percentage. To generate the testing set, 
            the data_idx generated by the train_set should be passed to the class 
            constructor of the test_set. 
            complement_idx: generate the set from the complement of data_idx
        """
        self.cf = cf
        self.dir_bmp = cf.dataset_path_IFN
        self.dir_tru = cf.gt_path_IFN
        self.train = train  # training set or test set
        self.transform = transform
        self.word_id = []
        self.word_str = []
        self.phoc_word = []

        
        aux_phoc_word, aux_word_id, aux_word_str = load_ifnedit_data(cf)        
        len_data = len(aux_word_id)
        
        if len(data_idx) == 1:  # this is safe as the lowest is one, when nothing is passed
            np.random.seed(cf.rnd_seed_value)
            data_idx = np.sort(np.random.choice(len_data, 
                                                size=int(len_data * cf.split_percentage), replace=False) )
            
        if complement_idx:
            all_idx = np.arange(0, len_data)
            data_idx = np.sort( np.setdiff1d(all_idx, data_idx, assume_unique=False) )

        for idx in data_idx:
            # self.phoc_word.append(aux_phoc_word[idx])
            self.word_id.append(aux_word_id[idx])
            self.word_str.append(aux_word_str[idx])
        
        self.data_idx = data_idx
        self.weights = np.ones( len(data_idx), dtype = 'uint8' )
    
    def add_weights_of_words(self): # weights to balance the loss, if the data is unbalanced   
        N = len(self.word_str)
        wordfreq = [self.word_str.count(w) for w in self.word_str]
        weights = 1 - np.array(wordfreq, dtype = 'float32')/N        
        self.weights = weights
    
#    def add_weights(self, weights):
#        self.weights = weights # weights to be used to balance the data, as input to the loss

    def num_classes(self):
        if self.cf.task_type=='script_identification':
        #    return len(self.cf.Arabic_label)
            return len(self.cf.PHOC('بجدهو', self.cf)) # pasing 'dump' word to get the length
        else:
            # return len(self.phoc_word[0])
            return len(self.cf.PHOC('بجدهو', self.cf)) # pasing 'dump' word to get the length
        
        
    def __len__(self):
        return len(self.word_id)
       
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.dir_bmp, self.word_id[idx] + '.bmp')
        data = Image.open(img_name)
        if not(self.cf.H_ifn_scale ==0): # resizing just the height            
            new_w = int(data.size[0]*self.cf.H_ifn_scale/data.size[1])
            if new_w>self.cf.MAX_IMAGE_WIDTH: 
                new_w = self.cf.MAX_IMAGE_WIDTH
            data = data.resize( (new_w, self.cf.H_ifn_scale), Image.ANTIALIAS)
        
        maxG = data.getextrema() # [0] is the min, [1] is the max        
        if maxG[1]>200: # correcting the values of folder e, they do not match the other folders
            data = data.point(lambda p: 0 if p == 255  else 255 )         
            ''' set_e has max of 255, while other sets, namely a,b,c,d have max of 1,
            abcd however need inversion, so, the one  line below works for all,
            to check each dataset use data.show() 
            if self.cf.IFN_test == 'set_e': '''
            
        else:
            data = data.point(lambda p: 1 if p == 0  else 0 ) # inverting and normalizing set_e to 1               
            # stragnely, this 
                
        data = data.convert('1')  # needs to be done to have all datasets in the same mode
        
        word_str = self.word_str[idx]
        if self.transform:
            data = self.transform(data)
        
        
        if self.cf.task_type=='script_identification':
            # target = self.cf.Arabic_label #  lable for Arabic script
            target = self.cf.PHOC('عربى9231', self.cf) # Arabic + hashcode
        else:
            # target = self.phoc_word[idx]
            target = self.cf.PHOC(word_str, self.cf)

        return data, target, word_str, self.weights[idx]
    
   
   
#  data = data.point(lambda p: 1 if p < 127  else 0 ) # threshold and invert            
