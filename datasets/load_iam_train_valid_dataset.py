#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 10:59:35 2018

@author: malrawi
"""

import torch.utils.data as data
from datasets.load_iam_dataset import IAM_words

'''
- Args
 
==
'''
# This can be used to only IFN data from four folders
class iam_train_valid_combined_dataset(data.Dataset):
  def __init__(self, cf, train=True, transform=None):
      
    # cf.train_split = False # this should always be false, as we are keeping one folder for testing
    self.train = train  # training set or test set      
    
    self.train_set = IAM_words(cf, mode='train', transform = transform)
    self.validate_set = IAM_words(cf, mode='validate', transform = transform)
    # backing up the original paths    
    
    self.train_set_len = len(self.train_set)
    self.validte_set_len = len(self.validate_set)
           
  
  def __getitem__(self, index):
    if index < self.train_set_len:
        return self.train_set[index]
    
    else:
        index = index - self.train_set_len
        return self.validate_set[index]  # check: are we skipping a sample here? 

  def __len__(self):
      return self.train_set_len + self.validte_set_len


  def add_weights_of_words(self): # weights to balance the loss, if the data is unbalanced   
      self.train_set.add_weights_of_words()

  def num_classes(self):
      return self.train_set.num_classes() # Does not matter which one as they all have the same phoc length
