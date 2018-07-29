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
data_set = IfnEnitDataset(cf, train=True, transform=None)
# data_set = WashingtonDataset(cf, train=True, transform=None)
# find_max_HW_in_data(data_set)


