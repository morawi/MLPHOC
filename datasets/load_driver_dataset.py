#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 20:24:48 2018

@author: malrawi
"""


import os
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

''' Kaggle Safe Driver dataset'''

# Class labels
labels = {  'c0' : 'safe driving', 
            'c1' : 'texting - right', 
            'c2' : 'talking on the phone - right', 
            'c3' : 'texting - left', 
            'c4' : 'talking on the phone - left', 
            'c5' : 'operating the radio', 
            'c6' : 'drinking', 
            'c7' : 'reaching behind', 
            'c8' : 'hair and makeup', 
            'c9' : 'talking to passenger'}

# Class labels
classes = ['safe.driving-19',  # As HashoingCode, to prevent collision with similar words. we used . and - as they are defined in the Gw_unigrams
            'texting.right-74', 
            'talking-phone-right', # 'talking.on.the-phone-right'
            'texting.left-73', 
            'talking-phone-left', 
            'operating.the-radio', 
            'drinking-481', 
            'reaching.behind', 
            'hair.and-makeup', 
            'talking.to-passenger']


class SafeDriverDataset(Dataset):
    """
    Arguments:
        cf: contains a few parameters, like split_percentage (70% for train 
        and 30% for testing); the path to train folder, etc, see config_file_wg.py
        transform: Torch composite transfrom to be perfomed on each item of get_item method
        data_idx: if 1, this means get data with split_percentage, else, 
        one can enter the indices of what samples not to choose
        For example, after constructing the training set, one can use
        train_set.data_idx as input to the testing set constructor to 
        get the complement (test) set
        
        **** The indices of Validation and Train dataset are shuffled****
        
    """

    def __init__(self, cf,  train=True, transform=None, data_idx = np.arange(1)):
    
        self.cf=cf
        self.Train = train
        self.transform = transform                
        
        # reading file names and labels
        all_file_names= []; all_labels = []
        for i, label in enumerate(labels):
            path_folder = os.path.join(cf.safe_driver_path, label, '*.jpg')
            files = glob.glob(path_folder) 
            all_file_names.extend(files)
            all_labels.extend([int(label[-1])]*len(files))
        
        len_data = len(all_file_names) #length
        
        if len(data_idx) == 1:  # this is safe as the lowest is one, when nothing is passed
            np.random.seed(cf.rnd_seed_value)
            self.data_idx = np.sort(np.random.choice(len_data, size=int(len_data * cf.split_percentage), 
                                                replace=False) )
        else:        
            all_idx = np.arange(0, len_data)
            self.data_idx = np.sort( np.setdiff1d(all_idx, data_idx, assume_unique=False) )

       
        self.img_names = [all_file_names[i] for i in self.data_idx]
        self.label_names = [all_labels[i] for i in self.data_idx]


    def __len__(self):
        return len(self.img_names)
    
    def num_classes(self):          
        return len(self.cf.PHOC('dump', self.cf)) # pasing 'dump' word to get the length
    
    def __getitem__(self, index):
        img_name = self.img_names[index]
        label = self.label_names[index]
        image = Image.open(img_name)          
        
        if not(self.cf.H_sfDrive_scale ==0): # resizing just the height            
            new_w = int(image.size[0]*self.cf.H_sfDrive_scale/image.size[1])
            if new_w>self.cf.MAX_IMAGE_WIDTH: 
                new_w = self.cf.MAX_IMAGE_WIDTH
            image = image.resize( (new_w, self.cf.H_sfDrive_scale), Image.ANTIALIAS)
        
        if self.transform is not None:
            image = self.transform(image)
        word_str = classes[label]
        target = torch.from_numpy(self.cf.PHOC(word_str, self.cf))
        return image, target, word_str, 0 # I am returning zero weights, as they are not useful
