#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 18:18:18 2018

@author: malrawi
"""
import numpy as np
import pandas as pd
from utils.retrieval import map_from_query_test_feature_matrices, map_from_feature_matrix
import torch

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

    
def remove_single_words(word_str):
    # find the locations of all single 'appearance word'    
    loc =  (pd.Series(word_str).duplicated(keep=False)).astype(int).tolist()
#   Removing all non-duplicates:
    s = pd.Series(word_str)
    word_str =  s[s.duplicated(keep=False)].tolist()    
    return word_str, loc



def find_mAP(word_str, pred, target, metric):
    # for tyeps fo metric, see retrieval.py
    # remove single wors from pred and target all
    word_str, loc = remove_single_words(word_str)
    loc = torch.ByteTensor(loc)
    pred = pred[loc]  # we have to negate loc
    target = target[loc]        
    
    query_labels = word_to_label(word_str)                 
    mAP_QbE, avg_precs = map_from_query_test_feature_matrices(target, pred, 
                                                          query_labels, query_labels,  metric)
    mAP_QbS, avg_precs = map_from_feature_matrix(pred,query_labels,'cosine', False)
    
    return mAP_QbE,mAP_QbS
    