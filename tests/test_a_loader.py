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
from scripts.data_transformations import ImageThinning, image_thinning, TheAugmentor
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
    #  img.show() # img show has a problem!!!!!!!!!!! IFN showed this
    # img = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 1) # this line does inversion of the image, IFN abcd, donno why?!
#    
    img_max= img.getextrema()[1]
    print('max gray val is: ', img_max)
    for i in range(len(data_set)):    
        img = data_set[i][0].getdata() 
        img_sum = sum(img)  
        hh.append(img_sum /( img.size[1]*img.size[0] ))
    #plt.hist(img.squeeze()) # to plot the hist of an image, rough one
    plt.figure(" ")
    plt.hist(hh)    
    plt.xticks(np.arange(0, 1.1, .2))    
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

thin_image = ImageThinning(p = 0.25)
the_augmentor = TheAugmentor(probability=1, grid_width=8, grid_height=3, magnitude=8)
# p.shear(probability=1, max_shear_left=10, max_shear_right=10)
sheer_tsfm = transforms.RandomAffine(0, shear=(-30,10) )
random_sheer = transforms.RandomApply([sheer_tsfm], p=0.7)
image_transfrom = transforms.Compose([thin_image,
                                      the_augmentor, 
                                     # sheer_tsfm, 
                                    #  transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                         # transforms.ToPILImage(),                                     
                         # transforms.ToTensor(),
                         # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                         # transforms.Normalize(mean, std),
                          ])

image_transfrom = None

if cf.dataset_name == 'IFN':       
    test_set = IfnEnitDataset(cf, train=True, transform = image_transfrom)

elif cf.dataset_name == 'IAM':    
    # test_set  = IAM_words(cf, mode='validate', transform = None) #image_transfrom)
    test_set  = IAM_words(cf, mode='test', transform = image_transfrom)
    
elif cf.dataset_name == 'WG':
    test_set = WashingtonDataset(cf, train=True, transform = image_transfrom)               
else: 
    print(' incorrect dataset_name')
    
x1 = test_set[921][0]
# x1 = x1.convert('L')
# plt.imshow(np.array(x1).squeeze(), 'gray')
# data_set  = IAM_words(cf, mode='test', transform = None)
#x2, _,_ = data_set2[1]

#find_max_HW_in_data(test_set)
# hist_of_text_to_background_ratio(test_set)
# test_thinning(test_set)






