#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:35:06 2019

@author: malrawi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 01:55:19 2018

@author: malrawi
"""

from scripts.Word2PHOC import build_phoc as build_phoc

def var_phoc(word, cf):
    phoc = build_phoc(word, cf)
    m =  len(word) # n = word_phoc_level
    word_phoc_size = len(cf.phoc_unigrams)*(m*(m+1)/2-1) 
    phoc[int(word_phoc_size):] = 0   

            
    return phoc



    
    