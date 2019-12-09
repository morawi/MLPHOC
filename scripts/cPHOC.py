#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:45:40 2019

@author: malrawi
"""

# from scripts.Word2PHOC import build_phoc as get_phoc

# Class PHOC
class cPHOC(object):

    def __init__(self, phoc, n_levels, word):
        self.n_levels = n_levels
        self.word = word
        self.phoc = phoc
        
        