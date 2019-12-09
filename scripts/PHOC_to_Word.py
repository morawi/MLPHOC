#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:55:20 2019

@author: malrawi

The PHOC decoder

The PHOC is contains concatnated vectors of the levels:
    Level2 has two parts: L2_1 and L2_2, 
    Level3 has three parts: L3_1, L3_2, and L3_3
    
    When doing analysis with levels 2 and 3, the PHOC thus will be represented by:
        L2_1|L2_2|L3_1|L3_2|L3_3
    For English words, 1% has more than 17 letters in a word. 
    Thus, to build an efficient decoder, we can use levels 8, 9, 10, and 11.
    To give an example, Level 8 implies ividing a 16 letter word into eight regions,
    thus, it is less likely that two letters in each region are the same. 
    
    The idea behined the decoder is to build an emsemble from the PHOC-levels[8,9,10,11].
    One can, of course, use higher levels to capture more word lengths, and to to minimize
    the probability of having same letters in one region, but that has to be on the expense of 
    model complexity. For English alphabets+numerals, the size is 36; hence, the total PHOC 
    length using PHOC-levels[8,9,10,11] will be  36*38= 1,368.

"""
def rotate(strg, n):
    return strg[n:] + strg[:n]

rotate('HELLO', -1)  # 'OHELL'

in_word = 'dcba'

total_n = sum(train_set.cf.unigram_levels)

for n in range(total_n):
    sz = len(train_set.cf.phoc_unigrams); 
    xx= train_set.cf.PHOC(in_word, train_set.cf); 
    print(xx[n*sz:n*sz+sz]); len(xx[n*sz:n*sz+sz])