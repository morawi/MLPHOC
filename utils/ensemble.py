#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:08:12 2019

@author: malrawi
"""


import numpy as np
from math import floor

class PHOC_ensemble():
    def __init__(self, cf, ensemble_type = 'average', phoc_vectors): # phoc_vectors  have a size n_test_samplesXn_ensmbles
        


        

#def bincount2D_vectorized(a):    
#    N = a.max()+1
#    a_offs = a + np.arange(a.shape[0])[:,None]*N
#    return np.bincount(a_offs.ravel(), minlength=a.shape[0]*N).reshape(-1,N)
#
