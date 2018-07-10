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

"""Example:
    x= ['hello', 'John', 'hi', 'John', 'hello', 'pumpum']
    output should be something like this:
    y=[0, 1, 2, 1, 0, 3] """
def word_to_label(word_str):
    d = {}
    
    count = 0
    for i in word_str:
      if i not in d:
         d[i] = count
         count += 1
    
    labels = [d[i] for i in word_str]
    print("Testing has", len(np.unique(labels)), " unique words out of", len(labels) )
    return labels
 
    
"""
Example:
  x=['hi', 'xerox', 'hi', 'xerox', 'dunk', 'hi']
then,
    word_str  = ['hi', 'xerox', 'hi', 'xerox', 'hi']  
    loc = [1, 1, 1, 1, 0, 1]
"""    
def remove_single_words(word_str):
    # find the locations of all single 'appearance word'    
    loc =  (pd.Series(word_str).duplicated(keep=False)).astype(int).tolist()
#   Removing all non-duplicates:
    s = pd.Series(word_str)
    word_str =  s[s.duplicated(keep=False)].tolist()    
    return word_str, loc



def find_mAP(word_str, pred, target, metric):
    # for tyeps of metric, see retrieval.py, or test_various_dist() shown below
    # remove single wors from pred and target all
    word_str, loc = remove_single_words(word_str)
    loc = torch.ByteTensor(loc)
    pred = pred[loc]  # we have to negate loc
    target = target[loc]        
    
    query_labels = word_to_label(word_str)                 
    mAP_QbE, avg_precs = map_from_query_test_feature_matrices(target, pred, 
                                                          query_labels, query_labels,  metric)
    mAP_QbS, avg_precs = map_from_feature_matrix(pred,query_labels, metric, False)
    
    return mAP_QbE,mAP_QbS
    



def test_varoius_dist(result):    
    all_distances = [ 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 
    'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',  
    'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 
    'sokalmichener', 'sokalsneath', 'sqeuclidean',  'yule']
    
    # 'mahalanobis' removed  due to Singular matrix
    # 'wminkowski': requires a weighting vector
    for my_distance in all_distances:
         mAP_QbE,mAP_QbS = find_mAP(result['word_str_all'],
                                    result['pred_all'],
                                    result['target_all'],
                                    my_distance)
         print('using', my_distance, '---- mAP(QbS)=', mAP_QbS, "---", 
               'mAP(QbE) = ', mAP_QbE, '----\n')
        
    
    















