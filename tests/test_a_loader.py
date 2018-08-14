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
from scripts.data_transformations import ImageThinning, image_thinning
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as matplot_rc
import torchvision.transforms as transforms

def find_max_HW_in_data(data_set):
    max_w=0; max_h = 0
    for  i in range(len(data_set)):
        if not i%991: print(i,',', end='')
        h, w, b = data_set[i][0].shape
        max_w = max(max_w, w)
        max_h = max(max_h, h)
        
    print('\n max_h', max_h)
    print('max_w',max_w) 
    
    

def hist_of_text_to_background_ratio(data_set):
    '''
    if the final histogram of the words is skewed to the right, 
    then, uncomment img=max_gray-img 
    '''
    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 16}
    matplot_rc('font', **font)
    hh = []    
    img = data_set[0][0]; 
    img_max = np.array(img).max()
    print('max gray val is: ', img_max)
    for i in range(len(data_set)):    
        img = np.array(data_set[i][0])
        hh.append(img.sum()/(img.size*img_max))
    # plt.hist(img.squeeze()) # to plot the hist of an image, rough one
    plt.figure(" ")
    plt.hist(hh)
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.show()


def test_thinning(data_set):
    img = data_set[45][0]; # img.sum()/(img.size*255)
    img1 = image_thinning(img, .1)
    plt.imshow(np.array(img).squeeze(), 'gray')
    plt.imshow(np.array(img1).squeeze(), 'gray')


config_path = 'config/config_file_wg.py'
configuration = Configuration(config_path, '')
cf = configuration.load()
cf.split_percentage = 1


folder_of_data                = '/home/malrawi/Desktop/My Programs/all_data/'
dataset_path_IFN              = folder_of_data + 'ifnenit_v2.0p1e/data/set_d/bmp/' # path to IFN images
gt_path_IFN                   = folder_of_data + 'ifnenit_v2.0p1e/data/set_d/tru/' # path to IFN ground_truth 
cf.dataset_path_IFN = dataset_path_IFN
cf.gt_path_IFN = gt_path_IFN


thin_image = ImageThinning(p = cf.thinning_threshold)
image_transfrom = transforms.Compose([ thin_image,
                         transforms.ToPILImage(),
                         transforms.ToTensor(),
                         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                         # transforms.Normalize(mean, std),
                          ])

cf.dataset_name = 'IAM'; 
# data_set  = IAM_words(cf, mode='validate', transform = None) #image_transfrom)
data_set  = IAM_words(cf, mode='validate', transform = image_transfrom)
x1 = data_set[9][0]
# data_set  = IAM_words(cf, mode='test', transform = None)
#x2, _,_ = data_set2[1]
#cf.dataset_name = 'WG'
# data_set = WashingtonDataset(cf, train=True, transform=image_transfrom)
#cf.dataset_name                 = 'IFN'   ; cf.H_ifn_scale = 0
#data_set = IfnEnitDataset(cf, train=True, transform=None)

# find_max_HW_in_data(data_set)
hist_of_text_to_background_ratio(data_set)

# test_thinning(data_set)





