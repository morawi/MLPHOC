#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 01:55:19 2018

@author: malrawi
"""
import numpy as np
from scripts.Word2PHOC import build_phoc as build_phoc
import sys

def build_rawhoc(word, cf):
    no_symbols = len(cf.phoc_unigrams)
    bitword_width = np.ceil(np.log2(no_symbols)).astype('uint8')
    rawhoc = np.zeros(cf.rawhoc_repeates*bitword_width*cf.max_word_len)

    for index, char in enumerate(word):
        i = cf.phoc_unigrams.find(char)  # the one added to get rid of zero binary
        if i < 0: print(char, 'not in unigrams '); sys.exit('Quiting')
        bb = bin(i+1); bb=bb[2:]
        bb = '0'*(bitword_width-len(bb))+bb  
        bb = list(bb*cf.rawhoc_repeates)
        bb = np.array(bb, dtype='uint8')
        offset = index*cf.rawhoc_repeates*bitword_width
        rawhoc[offset: offset + cf.rawhoc_repeates*bitword_width] = bb
    
    # rawhoc = [1-cf.tolerance if x>0.5 else cf.tolerance for x in rawhoc]
    return rawhoc 


def build_pro_hoc(word, cf):
    rawphoc =  build_rawhoc(word, cf)
    phoc = build_phoc(word, cf)
    p_raw_hoc = np.append(phoc, rawphoc)
    return p_raw_hoc
    
    