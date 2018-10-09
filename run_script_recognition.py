#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 14:21:13 2018

@author: malrawi
"""

from __future__ import division
import time
import logging

from config.load_config_file import Configuration
from tests.test_script_recognition import script_recognition
from utils.some_functions import random_seeding
from inspect import getmembers

import sys
import calendar as cal
import datetime
# from tests.test_dataset import test_dataload


start_timer = time.time()
test_name = '' # Optional: name of the sub folder where to store the results

logger = logging.getLogger(__name__)
# Load the configuration file
configuration = Configuration('config/config_file_wg.py', test_name)
cf = configuration.load()

# redirect printed results to ouptut file
if cf.redirect_std_to_file:    
    dd = datetime.datetime.now()
    out_file_name = cf.dataset_name +'_'+ cal.month_abbr[dd.month]+ '_' + str(dd.day)
    print('Output sent to ', out_file_name)
    sys.stdout = open(out_file_name,  'w')

# print all the parameters
xx = getmembers(cf)
for i in range(len(xx)): 
    print (xx[i])
print('\n --------------- Script identification: English(0) vs Arabic(1) ----------')
if cf.overlay_handwritting_on_STL_img:
    print('------ Scene Handwritting Experiment ----------')

random_seeding(seed_value = cf.rnd_seed_value, use_cuda=True) # randomizer

result, train_set, test_set, train_loader, test_loader = script_recognition(cf) # Test CNN finetune with WG dataset    

print("Execution time is: ", time.time() - start_timer )
    





