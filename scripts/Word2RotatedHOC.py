#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 14:03:42 2019

@author: malrawi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 01:55:19 2018

@author: malrawi
"""
import numpy as np
from scripts.Word2PHOC import build_phoc as build_phoc

def rotate_string(strg, n):
    return strg[n:] + strg[:n]


def rotated_hoc(word, cf):
    rotated_hoc = np.array(0, dtype= 'float32')
    for n in range(cf.no_word_rotations):
        phoc = build_phoc( rotate_string(word, n), cf)
        rotated_hoc = np.append(rotated_hoc, phoc)
    return rotated_hoc
    
    