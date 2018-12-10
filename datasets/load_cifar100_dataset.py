#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 14:17:15 2018

@author: malrawi
"""

import torchvision
from torch.utils.data import Dataset
import numpy as np

def load_cifar100_dataset(cf, mode):
    if mode == 'train':
        data_set = torchvision.datasets.CIFAR100(
                    root = cf.cifar100_path, train=True, download=True, transform=None)
    else:
        data_set = torchvision.datasets.CIFAR100(
                    root=cf.cifar100_path, train=False, download=True, transform=None)
            
    # I added numbers in case some names appear in handwriting or text detection datasets    
    classes = (
       'apple123', 'aquarium-fish45', 'baby456', 'bear890', 'beaver149', 'bed891', 'bee432', 
       'beetle675', 'bicycle213', 'bottle576', 'bowl912', 'boy135', 'bridge843', 'bus582', 
        'butterfly370', 'camel431', 'can942', 'castle638', 'caterpillar61', 'cattle936', 
        'chair231', 'chimpanzee954', 'clock953', 'cloud206', 'cockroach037', 'couch126', 
        'crab947', 'crocodile732', 'cup1426', 'dinosaur093', 'dolphin098', 'elephant765', 
        'flatfish432', 'forest109', 'fox876', 'girl5439',  'hamster543', 'house210', 
        'kangaroo912', 'keyboard834', 'lamp756', 'lawn-mower049', 'leopard867', 
        'lion086', 'lizard531', 'lobster975', 'man420', 'maple-tree135', 'motorcycle246', 
        'mountain791', 'mouse124', 'mushroom245', 'oak-tree568', 'orange932', 'orchid710', 
        'otter364', 'palm-tree752', 'pear0491', 'pickup-truck305', 'pine-tree486', 'plain993', 'plate749', 
        'poppy350', 'porcupine372', 'possum736', 'rabbit274', 'raccoon912', 'ray836', 
        'road632', 'rocket048', 'rose320', 'sea1478', 'seal9634', 'shark9634', 'shrew5306', 
        'skunk13057', 'skyscraper91', 'snail90865', 'snake08231', 'spider37629',
        'squirrel873', 'streetcar3261', 'sunflower716', 'sweet-pepper549', 'table74821',
        'tank10467', 'telephone9372', 'television710', 'tiger76302', 'tractor9365',
        'train3201', 'trout7392', 'tulip14820', 'turtle6354', 'wardrobe2745',
        'whale7392', 'willow-tree193', 'wolf64532', 'woman67320', 'worm7392'
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
        img = img.resize( (self.cf.w_new_size, self.cf.h_new_size)) # zooming in is magic for cifar100
        if self.transform:
            img = self.transform(img)
        class_id = self.dataset[idx][1]
        word_str = self.classes[ class_id ]           
        target = self.cf.PHOC(word_str, self.cf)
        return img, target, word_str, self.weights[idx]
       
           

            
            
            
           