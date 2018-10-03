#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:30:47 2018

@author: malrawi
"""

"""
dataset fusion based on:
    https://github.com/xingyizhou/pytorch-pose-hg-3d/blob/master/src/datasets/fusion.py """

import torch.utils.data as data
from datasets.load_ifnenit_dataset import IfnEnitDataset
from datasets.load_iam_dataset import IAM_words
import numpy as np

'''
- Args
First, load the train set by the default values, except (train and transform), 
this wil result in train_set, where the idx's are stored for each IFN and WG, 
then, to load the test set, these idx's are estimated randomly according to 
cf.split_percentage in config_file_wg.py. 
 We then can use these idx's (train_set.data_idx_WG,) to load the complement data and get 
the test set for each of IFN and IAM.

Example:
train_set = IAM_IFN_Dataset(cf, train=True, mode='train', transform=image_transfrom)
test_set = IAM_IFN_Dataset(cf, train=False, mode='test or validate' transform=image_transfrom, 
                      data_idx_WG = train_set.data_idx_WG, 
                      data_idx_IAM = train_set.data_idx_IFN, 
                            complement_idx = True)
 
==
'''
class IAM_IFN_Dataset(data.Dataset):
  def __init__(self, cf, train=True, mode = 'train', transform=None, 
               data_idx_IAM = np.arange(1), 
               data_idx_IFN = np.arange(1), 
               complement_idx=False):
    self.train = train  # training set or test set 
    self.mode = mode
    if len(data_idx_IFN)==1:
        self.datasetIFN = IfnEnitDataset(cf, train=self.train, transform = transform)
    else:
        self.datasetIFN = IfnEnitDataset(cf, train=self.train, transform = transform,
                                         data_idx = data_idx_IFN, complement_idx = True)
    if len(data_idx_IAM)==1:                
        self.datasetIAM = IAM_words(cf, mode = self.mode, transform = transform)                
    else:
        self.datasetIAM = IAM_words(cf, mode = self.mode, transform = transform)

    self.data_idx_IFN = self.datasetIFN.data_idx # this is needed, to be passed from one set to another
   #  self.data_idx_IAM = self.datasetIAM.data_idx # this is needed, to be passed from one set to another
          
  def add_weights_of_words(self): # weights to balance the loss, if the data is unbalanced   
      self.datasetIFN.add_weights_of_words()
      self.datasetIAM.add_weights_of_words()

  def num_classes(self):
    return self.datasetIAM.num_classes() #IFN and WG have the same phoc size

  def __getitem__(self, index):
      
    if index < len(self.datasetIFN):
        return self.datasetIFN[index]
    else:
        return self.datasetIAM[index - len(self.datasetIFN)] # check: are we skipping a sample here?

  def __len__(self):
    return len(self.datasetIFN) + len(self.datasetIAM)
