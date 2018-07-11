#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:32:18 2018

@author: malrawi
"""

"""
dataset fusion based on:
    https://github.com/xingyizhou/pytorch-pose-hg-3d/blob/master/src/datasets/fusion.py """

import torch.utils.data as data
from datasets.load_washington_dataset import WashingtonDataset
from datasets.load_ifnenit_dataset import IfnEnitDataset
import numpy as np


class WG_IFN_Dataset(data.Dataset):
  def __init__(self, cf, train=True, transform=None, data_idx_WG = np.arange(1), 
               data_idx_IFN = np.arange(1), complement_idx=False):
    self.train = train  # training set or test set  
    if len(data_idx_WG)==1:
        self.datasetWG = WashingtonDataset(cf, train=self.train, transform = transform)        
    else: 
        self.datasetWG = WashingtonDataset(cf, train=self.train, transform = transform,
                                           data_idx = data_idx_WG, complement_idx = True)
    if len(data_idx_IFN)==1:
        self.datasetIFN = IfnEnitDataset(cf, train=self.train, transform = transform)
    else:
        self.datasetIFN = IfnEnitDataset(cf, train=self.train, transform = transform,
                                         data_idx = data_idx_IFN, complement_idx = True)

    self.data_idx_WG = self.datasetWG.data_idx # this is needed, to be passed from one set to another
    self.data_idx_IFN = self.datasetIFN.data_idx # this is needed, to be passed from one set to another
        

  def num_classes(self):
    return self.datasetIFN.len_phoc

  def __getitem__(self, index):
    if index < len(self.datasetWG):
        return self.datasetWG[index]
    else:
        return self.datasetIFN[index - len(self.datasetWG)] # check: are we skipping a sample here?

  def __len__(self):
    return len(self.datasetWG) + len(self.datasetIFN)



