#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 18:11:50 2018

@author: malrawi

"""

import os
os.chdir("..") # bringing the directory back to MLPHOC

from config.load_config_file import Configuration
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as matplot_rc
import torch

from datasets.get_datasets import get_datasets,  get_dataloaders, get_transforms

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
   
    # img = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 1) # this line does inversion of the image, IFN abcd, donno why?!
#    
    img_max= img.getextrema()[1]
    print('max gray val is: ', img_max)
    for i in range(len(data_set)):    
        img = data_set[i][0].getdata() 
        img_sum = sum(img)  
        hh.append(img_sum /(img_max*img.size[1]*img.size[0] ))
    #plt.hist(img.squeeze()) # to plot the hist of an image, rough one
    plt.figure(" ")
    plt.hist(hh, 100)    
    plt.xticks(np.arange(0, 1.1, .2))    
    plt.show()


def test_thinning(data_set):
    img = data_set[45][0]; # img.sum()/(img.size*255)
    img1 = image_thinning(img, .1)
    plt.imshow(np.array(img).squeeze(), 'gray')
    plt.imshow(np.array(img1).squeeze(), 'gray')




configuration = Configuration(config_path='config/config_file_wg.py', test_name='')
cf = configuration.load()
image_transform = get_transforms(cf)
train_set, test_set, test_per_data = get_datasets(cf, image_transform)
train_loader, test_loader, per_data_loader = get_dataloaders(cf, train_set, test_set, test_per_data)

# id  = int( torch.randint(500, (1,1)).numpy()[0][0]); 
# x = test_per_data['test_Instagram'][id][0]

#if type(x) == torch.Tensor: 
#    plt.imshow(x[1,:])
#    plt.show()



# train_loader, test_loader, per_data_loader = get_dataloaders(cf, train_set, test_set, test_per_data)       
           
#train_set.collect_phoc_unigrams(['Bangla'])


## build the model
#model = make_model(
#    cf.model_name,
#    pretrained = cf.pretrained,
#    num_classes = 10, #train_set.num_classes(),
#    input_size = cf.input_size, 
#    dropout_p = cf.dropout_probability,
#)
#
## changing dropout value 
#model.dropout.p = .3



# x1 = test_set[561]
# plt.imshow(np.array(x1).squeeze(), 'gray')
# hist_of_text_to_background_ratio(test_set)
# x1 = x1.convert('L')
# plt.imshow(np.array(x1).squeeze(), 'gray')
# data_set  = IAM_words(cf, mode='test', transform = None)
#x2, _,_ = data_set2[1]

#find_max_HW_in_data(test_set)

# test_thinning(test_set)






