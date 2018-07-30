#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 18:11:50 2018

@author: malrawi
"""

import os

os.chdir("..")

from config.load_config_file import Configuration
from datasets.load_ifnenit_dataset import IfnEnitDataset
from datasets.load_washington_dataset import WashingtonDataset
from datasets.load_iam_dataset import IAM_words
from scripts.data_transformations import ImageThinning
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as matplot_rc

def find_max_HW_in_data(data_set):
    max_w=0; max_h = 0
    for  i in range(len(data_set)):
        if not i%991: print(i,',', end='')
        h, w, b = data_set[i][0].shape
        max_w = max(max_w, w)
        max_h = max(max_h, h)
        
    print('max_h', max_h)
    print('max_w',max_w)
    
    
    

config_path = 'config/config_file_wg.py'
configuration = Configuration(config_path, '')
cf = configuration.load()
cf.split_percentage = 1


folder_of_data              = '/home/malrawi/Desktop/My Programs/all_data/'
dataset_path_IFN              = folder_of_data + 'ifnenit_v2.0p1e/data/set_e/bmp/' # path to IFN images
gt_path_IFN                   = folder_of_data + 'ifnenit_v2.0p1e/data/set_e/tru/' # path to IFN ground_truth 
cf.dataset_path_IFN = dataset_path_IFN
cf.gt_path_IFN = gt_path_IFN


# data_set = IfnEnitDataset(cf, train=True, transform=None)
# data_set = WashingtonDataset(cf, train=True, transform=None)
cf.dataset_name = 'IAM'; data_set  = IAM_words(cf, mode='validate', transform = None, augmentation=True )
# thin_img = ImageThinning(img)

# find_max_HW_in_data(data_set)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplot_rc('font', **font)
hh = []

img, _, _ = data_set[0]; img_max = img.max()
print('max gray val is: ', img_max)
for i in range(len(data_set)):    
    img, tt, ss = data_set[i]
    # img = img_max  - img 
    hh.append(img.sum()/(img.size* img_max))
# plt.hist(img.squeeze()) # to plot the hist of an image, rough one
plt.hist(hh)





