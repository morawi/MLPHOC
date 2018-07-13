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
from collections import Counter
from nltk.corpus import wordnet


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
then, removing 'dunk'
    word_str  = ['hi', 'xerox', 'hi', 'xerox', 'hi']  
    loc = [1, 1, 1, 1, 0, 1]
"""    
def remove_single_words(word_str):
    # find the locations of all single 'appearance word'    
    loc =  (pd.Series(word_str).duplicated(keep=False)).astype(int).tolist()
    loc = torch.ByteTensor(loc)
#   Removing all single word(s):
    s = pd.Series(word_str)
    word_str =  s[s.duplicated(keep=False)].tolist()    
    return word_str, loc



def find_mAP(result, cf):
    
    
    pred = result['pred']       # test_phocs       
    pred = binarize_the_output(pred, cf.binarizing_thresh)
    # For QbE, we have to remove each single occurence words, 
    # and the corresponding items in pred
    word_str = result['word_str']
    target = result['target'] 
    word_str, loc = remove_single_words(word_str)    
    pred = pred[loc]   
    target = target[loc]
    query_labels = word_to_label(word_str)                 
    mAP_QbE, avg_precs = map_from_feature_matrix(pred, query_labels, cf.mAP_dist_metric, False)
    
    # function_form is: map_from...matrices(query_features, test_features, query_labels, test_labels
    mAP_QbS, avg_precs = map_from_query_test_feature_matrices(target, pred, 
                                                          query_labels, query_labels, cf.mAP_dist_metric)
    
    '''   
    # For QbS, we have to remove the repeated word-embbedings in the target (i.e., query features)
    # we also have to remove the same items from the word string, so that we generate the target_labels (query_labels)
    word_str = result['word_str'] #  use the original one    
    target = result['target']  # query_phocs, unique for Qbs    
    # idx_unique = sorted(word_str.index(elem) for elem in set(word_str))
    idx_unique = [word_str.index(elem) for elem in set(word_str)]
    word_str = pd.Series(word_str)
    word_str = word_str.get(idx_unique)
    target_labels = word_to_label(word_str) 
    target = target[idx_unique]
    
    # function_form is: map_from...matrices(query_features, test_features, query_labels, test_labels
    mAP_QbS, avg_precs = map_from_query_test_feature_matrices(target, pred, 
                                                          target_labels, query_labels, cf.mAP_dist_metric)
    '''
    
    
    return mAP_QbE,mAP_QbS
    



def test_varoius_dist(result, cf):    
    all_distances = [ 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 
    'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',  
    'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 
    'sokalmichener', 'sokalsneath', 'sqeuclidean',  'yule']
    
    # 'mahalanobis' removed  due to Singular matrix
    # 'wminkowski': requires a weighting vector
    for my_distance in all_distances:
        cf.mAP_dist_metric = my_distance
        mAP_QbE,mAP_QbS = find_mAP(result, cf)
        print('using', my_distance, '---- mAP(QbS)=', mAP_QbS, "---", 
               'mAP(QbE) = ', mAP_QbE, '----\n')

    
    
def binarize_the_output(pred, binarizing_thresh):        
    if (binarizing_thresh ==0.5):
        pred = pred.round()
    else:        
        pred = ( pred > binarizing_thresh)
    return pred


def test_varoius_thresholds(result, cf):    
    thresholds = np.arange(1,20)/20
    
    # 'mahalanobis' removed  due to Singular matrix
    # 'wminkowski': requires a weighting vector
    for my_thresh in thresholds:
        cf.binarizing_thresh = my_thresh
        mAP_QbE,mAP_QbS = find_mAP(result, cf)
        print('Thresh val', my_thresh, '---- mAP(QbS)=', mAP_QbS, "---", 'mAP(QbE) = ', mAP_QbE, '----\n')


def word_str_moment(word_str):
    word_str, loc = remove_single_words(word_str)
    vals, ids, idx = np.unique(word_str, return_index=True, return_inverse=True)
    vv= Counter(idx)
    ss0 = sum(vv.values())
    ss = 0    
    for i in range(0,  len(vv) ):
        ss += vv[i]**2
    
    ss= ss/( len(vv) * ss0)
    return ss
        
 
def word_similarity_metric(list_of_words):
    ''' Example:   list_of_wrods  = ['hi', 'xerox', 'hi', 'xerox', 'dunk', 'hi']    
    print( word_similarity_metric(list_of_wrods) )
    
    '''
    
    list_of_words = list(list_of_words)
    list_of_words, loc = remove_single_words(list_of_words)  
    ss=0
    list_len =  len(list_of_words)
    for word in list_of_words:
        list_of_words.pop(0)
        ss += ListOfWords_to_ListOfWords_statistic(list([word]), list_of_words )
    return 2*(ss/list_len)


def ListOfWords_to_ListOfWords_statistic(list1, list2):

# input args
# list1  ['choose', 'copy', 'define', 'copy', 'choose', 'choose']
# list2 has several words to be compared with the ones in lis1
    
    xx = []; i=0    
    for word1 in list1:
        for word2 in list2:
            wordFromList1 = wordnet.synsets(word1)
            wordFromList2 = wordnet.synsets(word2)
            if wordFromList1 and wordFromList2: 
                s = wordFromList1[0].wup_similarity(wordFromList2[0])
                xx.append(s)        
    for zz in xx:
         xx[i] = int(0 if zz is None else zz)
         i += 1
        
    return sum(xx)/( len(list1) * len(list2) )








