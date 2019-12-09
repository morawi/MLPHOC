#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 14:17:15 2018

@author: malrawi
"""

import torchvision
from torch.utils.data import Dataset
import numpy as np
import torch

from PIL import Image
from PIL import ImageEnhance 


def load_cifar100_dataset(cf, mode):
    if mode == 'train':
        data_set = torchvision.datasets.CIFAR100(
                    root = cf.cifar100_path, train=True, download=True, transform=None)
    else:
        data_set = torchvision.datasets.CIFAR100(
                    root=cf.cifar100_path, train=False, download=True, transform=None)
            
    # I added numbers in case some names appear in handwriting or text detection datasets    
    classes = (
       'apple', 'aquarium-fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 
       'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 
        'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 
        'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 
        'flatfish', 'forest', 'fox', 'girl',  'hamster', 'house', 
        'kangaroo', 'keyboard', 'lamp', 'lawn-mower', 'leopard', 
        'lion', 'lizard', 'lobster', 'man', 'maple-tree', 'motorcycle', 
        'mountain', 'mouse', 'mushroom', 'oak-tree', 'orange', 'orchid', 
        'otter', 'palm-tree', 'pear', 'pickup-truck', 'pine-tree', 'plain', 'plate', 
        'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 
        'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 
        'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet-pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe',
        'whale', 'willow-tree', 'wolf', 'woman', 'worm'
        )
    
    return data_set, classes

class Cifar100Dataset(Dataset):

    def __init__(self, cf, mode='train', transform = None):
        # mode: 'train', 'validate', or 'test'        
        self.cf = cf
        self.mode = mode        
        self.transform  = transform
        self.dataset, self.classes = load_cifar100_dataset(cf, mode)                      
        self.weights = np.ones( len(self.dataset) , dtype = 'uint8')        
        
    def num_classes(self):          
        return len(self.cf.PHOC('dump', self.cf)) # pasing 'dump' word to get the length
            
      
    def __len__(self):
        return len(self.dataset)                  
        
    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        if self.cf.w_new_size_cifar100>32:
            img = img.resize( (self.cf.w_new_size_cifar100, self.cf.h_new_size_cifar100)) # zooming in is magic for cifar100
       
        # img = ImageEnhance.Color(img).enhance(.8)  # https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
        if self.transform:
            img = self.transform(img)
        
        class_id = self.dataset[idx][1]
        word_str = self.classes[ class_id ]           
        target = torch.from_numpy( self.cf.PHOC(word_str, self.cf, mode='vision') )
        
        return img, target, word_str, self.weights[idx]
       
           

            
            
# img = Image.eval(img, lambda px: (px**1.1)%255)
           