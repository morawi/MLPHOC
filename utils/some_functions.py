#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 18:18:18 2018

@author: malrawi
"""
import numpy as np

def word_to_label(word_str):
    d = {}
    
    count = 0
    for i in word_str:
      if i not in d:
         d[i] = count
         count += 1
    
    labels = [d[i] for i in word_str]
    print("There are ", len(np.unique(labels)), " unique words" )
    return labels
    