#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:07:33 2019

@author: malrawi
"""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter

def annotation_exists(word, cf):        
    char_indices = {d: i for i, d in enumerate(cf.phoc_unigrams)}    
    for index, char in enumerate(word):        
        if char not in char_indices:
            print(char,' ', end='')
            return False
    return True

class MLT_words(Dataset):
    def __init__(self, cf,  train=True, transform=None, data_idx = np.arange(1)):
        self.cf = cf
        self.train = train        
        self.transform = transform   
        self.img_name = []
        self.language = []
        self.word = []        
        self.get_MLT_file_label()        
        self.no_word_per_language = Counter(self.language)
        len_data = len(self.img_name)
        if len(data_idx) == 1:  # this is safe as the lowest is one, when nothing is passed
            np.random.seed(cf.rnd_seed_value)
            self.data_idx = np.sort(np.random.choice(len_data, size=int(len_data * cf.split_MLT), 
                                                replace=False) )
        else:        
            all_idx = np.arange(0, len_data)
            self.data_idx = np.sort( np.setdiff1d(all_idx, data_idx, assume_unique=False) )
        
        self.img_name = [self.img_name[i] for i in self.data_idx]
        self.language = [self.language [i] for i in self.data_idx]
        self.word = [self.word[i] for i in self.data_idx]
        
        self.weights =  0 # depriciated for the time being; np.ones( len(self.file_label) , dtype = 'uint8' )               
    def get_MLT_file_label(self):                         
        gt_tr = 'gt.txt'
        print('Wrong char annotations will be printed out below........')
        with open(self.cf.dataset_path_MLT + gt_tr, 'r') as f_tr:
            data_tr = f_tr.readlines()
            for data_item in data_tr:  
                data_item = data_item.split(',') 
                if len(data_item) ==3: # some records have no language or error, like 2064
                    im_nm, ln, ws  = data_item                    
                    if ln in self.cf.MLT_language: #if ln=='English' or ln=='Arabic': # pick only two languages, English and Arabic
                        ws = ws[:-1] # taking out the new line symbol '\n'
                        if len(ws)<3: continue # removing small words or single characters
                        if ln=='English': ws = ws.lower()                                                           
                        if not annotation_exists(ws, self.cf): continue                    
                        self.img_name.append(im_nm)
                        self.language.append(ln)
                        self.word.append(ws)
                                                     
    
    def __getitem__(self, index):            
        img = Image.open(self.cf.dataset_path_MLT + 'MLT_images/' + self.img_name[index])        
        word_str = self.word[index]
        if not(self.cf.H_MLT_scale ==0): # resizing just the height             
            new_w = int(img.size[0]*self.cf.H_MLT_scale/img.size[1])
            if new_w>self.cf.MAX_IMAGE_WIDTH: 
                new_w = self.cf.MAX_IMAGE_WIDTH
            img = img.resize( (new_w, self.cf.H_MLT_scale), Image.ANTIALIAS)               
        target = self.cf.PHOC(word_str, cf = self.cf)    
        if img.mode !='RGB':
            img = img.convert('RGB')
                    
        if self.transform:
            img = self.transform(img)    
        
        return img, target, word_str, self.weights

    def __len__(self):
        return len(self.word)
     
    def num_classes(self):
        if self.cf.encoder=='label':
            return len(self.cf.English_label)
        else:
            return len(self.cf.PHOC('dump', self.cf)) # pasing 'dump' word to get the length


#
#configuration = Configuration(config_path='config/config_file_wg.py', test_name='') # test_name: Optional name of the sub folder where to store the results
#cf = configuration.load(print_vals=True)
#train_set = MLT_words(cf, train = True)
#train_set[1]
#test_set = MLT_words(cf, train = False)