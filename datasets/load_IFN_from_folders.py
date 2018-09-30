#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:58:22 2018

@author: malrawi

Loading IFN-ENIT four folders; training set
The test set can be loader with 
test_set = IfnEnitDataset(cf, train= False, transform = transform)
but cf.IFN_test should have the name of the test folder (e.g. set_a), 
which is skipped in IFN_XVAL_Dataset

dataset fusion based on:
    https://github.com/xingyizhou/pytorch-pose-hg-3d/blob/master/src/datasets/fusion.py """

import torch.utils.data as data
from datasets.load_ifnenit_dataset import IfnEnitDataset

'''
- Args
 
==
'''
# This can be used to only IFN data from four folders
class IFN_XVAL_Dataset(data.Dataset):
  def __init__(self, cf, train=True, transform=None):
      
    # cf.train_split = False # this should always be false, as we are keeping one folder for testing
    self.train = train  # training set or test set  
    
    trn_folder = cf.folders_to_use.replace(cf.IFN_test[-1], '') # removing the test set from train folders
    
    # backing up the original paths    
    dataset_path = cf.dataset_path_IFN
    gt_path = cf.gt_path_IFN 
    
    cf.dataset_path_IFN = dataset_path.replace(cf.IFN_test, 'set_'+ trn_folder[0] )
    cf.gt_path_IFN     =  gt_path.replace(cf.IFN_test, 'set_'+ trn_folder[0] )
    self.datasetIFN_1 = IfnEnitDataset(cf, train=self.train, transform = transform)
    
    cf.dataset_path_IFN = dataset_path.replace(cf.IFN_test, 'set_'+ trn_folder[1] )   
    cf.gt_path_IFN     =  gt_path.replace(cf.IFN_test, 'set_'+ trn_folder[1] )
    self.datasetIFN_2 = IfnEnitDataset(cf, train=self.train, transform = transform)
    
    cf.dataset_path_IFN = dataset_path.replace(cf.IFN_test, 'set_'+ trn_folder[2] )    
    cf.gt_path_IFN     =  gt_path.replace(cf.IFN_test, 'set_'+ trn_folder[2] )
    self.datasetIFN_3 = IfnEnitDataset(cf, train=self.train, transform = transform)
    
    cf.dataset_path_IFN = dataset_path.replace(cf.IFN_test, 'set_'+ trn_folder[3] )    
    cf.gt_path_IFN     =  gt_path.replace(cf.IFN_test, 'set_'+ trn_folder[3] )
    self.datasetIFN_4 = IfnEnitDataset(cf, train=self.train, transform = transform)
    
    self.IFN_1_len = len(self.datasetIFN_1)
    self.IFN_2_len = len(self.datasetIFN_2)
    self.IFN_3_len = len(self.datasetIFN_3)
    self.IFN_4_len = len(self.datasetIFN_4)
    
    cf.dataset_path_IFN = dataset_path
    cf.gt_path_IFN = gt_path   # this needs to be used in loading the test set
              
  
  def __getitem__(self, index):
    if index < self.IFN_1_len:
        return self.datasetIFN_1[index]
    
    elif index < (self.IFN_1_len + self.IFN_2_len):
        index = index - (self.IFN_1_len)
        return self.datasetIFN_2[index]  # check: are we skipping a sample here?
    
    elif index < (self.IFN_1_len + self.IFN_2_len + self.IFN_3_len):
        index = index - (self.IFN_1_len + self.IFN_2_len)
        return self.datasetIFN_3[index]  
    
    else: # This is IFN_4
        index = index - (self.IFN_1_len + self.IFN_2_len+ self.IFN_3_len )
       
        return self.datasetIFN_4[index]          
    

  def __len__(self):
      return self.IFN_1_len + self.IFN_2_len + self.IFN_3_len + self.IFN_4_len


  def add_weights_of_words(self): # weights to balance the loss, if the data is unbalanced   
      self.datasetIFN_1.add_weights_of_words()

  def num_classes(self):
      return self.datasetIFN_1.num_classes() # Does not matter which one as they all have the same phoc length
