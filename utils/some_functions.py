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
import random
from sklearn import preprocessing

# https://www.wordfrequency.info/comparison.asp



def random_seeding(seed_value, use_cuda):
    ''' a function to randomly seed torch and np
        Args in: seed_value: (int)
                 use_cuda: bool, True for using cuda or False otherwise
    '''

    np.random.seed(seed_value)
    torch.manual_seed(seed_value)   
    random.seed(seed_value)

    if use_cuda: torch.cuda.manual_seed_all(seed_value)
    
def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def word_to_label(word_str):
    """Example:
    x= ['hello', 'John', 'hi', 'John', 'hello', 'pumpum']
    output should be something like this:
    y=[0, 1, 2, 1, 0, 3] """

    d = {};  count = 0
   
    for i in word_str:
      if i not in d:
         d[i] = count
         count += 1
    
    labels = [d[i] for i in word_str]
   
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

    s = pd.Series(word_str)
    word_str =  s[s.duplicated(keep=False)].tolist()  #   Removing all single word(s):   
    return word_str, loc

def remove_stop_words(word_str) :
    # for stop_words list: https://gist.github.com/sebleier/554280 
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                  'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
                  'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                  "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
                  'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                  'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
                  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                  'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                  'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                  'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                  'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                  'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                  'again', 'further', 'then', 'once', 'here', 'there', 'when', 
                  'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 
                  'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
                  'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
                  'can', 'will', 'just', 'don', "don't", 'should', "should've", 
                  'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 
                  "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', 
                  "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', 
                  "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
                  'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
                  'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
                  'won', "won't", 'wouldn', "wouldn't"]
    non_alphanumeric = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/']
                        
    stop_words = stop_words + non_alphanumeric
                        
    word_str= list(word_str)
#    loc = []
#    for word in word_str:  # iterating on a copy since removing will mess things up        
#        if word in stop_words:
#            loc.append( word_str.index(word) )
#            word_str.remove(word)
 
    loc = [i for i, x in enumerate(word_str) if x not in stop_words]
    word_str = [word_str[i] for i in loc]
#     word_str = [x for x in word_str if x not in stop_words]
    
    return tuple(word_str), loc


def find_mAP_QbE(result, cf):
# For QbE, we have to remove each single occurence words, 
    # and the corresponding items in pred    
    pred = result['pred']       # test_phocs       
    pred = binarize_the_output(pred, cf.binarizing_thresh)      
    word_str = result['word_str']

    if cf.dataset_name=='IAM' or cf.dataset_name=='IAM+IFN':
        word_str, loc11 = remove_stop_words(word_str)
        pred = pred[loc11]
                
    query_labels, loc = remove_single_words(word_str)  
    pred = pred[loc]   
    query_labels = np.array(word_to_label(query_labels)).astype('uint32')
    mAP_QbE, avg_precs = map_from_feature_matrix(pred, query_labels, cf.mAP_dist_metric, False)    
    return mAP_QbE
          
    
'''
 get unique words are not correct,
we should extract unique phocs from pred
'''    
def find_mAP_QbS(result, cf):
    # For QbS, we have to use single (transcriptions) target phoc     
    # get unique phoc from target, target is the query 
    pred = result['pred']       # test_phocs 
    pred = binarize_the_output(pred, cf.binarizing_thresh)         
    word_str = result['word_str']      
            
    le = preprocessing.LabelEncoder()
    le.fit(word_str)
    pred_labels = le.transform(word_str).astype('uint32')
    
    target = result['target']
    if cf.dataset_name=='IAM' or cf.dataset_name=='IAM+IFN':
        word_str, loc11 = remove_stop_words(word_str)            
        target = target[loc11]
    
    target_labels, loc_unique = get_unique_words(word_str) # target is the query    
    target_labels = le.transform(target_labels).astype('uint32')    
    target = target[loc_unique]
    # function_form is: map_from...matrices(query_features, test_features, query_labels, test_labels
    
    mAP_QbS, avg_precs = map_from_query_test_feature_matrices(target, pred, 
                         target_labels, pred_labels, cf.mAP_dist_metric)
   
    return mAP_QbS


'''  Args in- list of words/strings 
ouptus: unique words and thier locations/positions'''
def get_unique_words(word_str):
    len_orig = len(word_str)
    loc = [word_str.index(elem) for elem in set(word_str)]
    word_str = pd.Series(word_str)
    word_str = word_str.get(loc)   
    print("There're", len(word_str), " unique words out of", len_orig, "  ", end = "" )
    return word_str, loc

def test_varoius_dist(result, cf):    
    all_distances = [ 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 
    'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',  
    'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 
    'sokalmichener', 'sokalsneath', 'sqeuclidean',  'yule']
    
    # 'mahalanobis' removed  due to Singular matrix
    # 'wminkowski': requires a weighting vector
    for my_distance in all_distances:
        cf.mAP_dist_metric = my_distance
        mAP_QbS = find_mAP_QbS(result, cf)
        mAP_QbE = find_mAP_QbE(result, cf)
      
        print('using', my_distance, '---- mAP(QbS)=', mAP_QbS, "---", 
               'mAP(QbE) = ', mAP_QbE, ' \n')
    
    
def binarize_the_output(pred, binarizing_thresh):        
    if (binarizing_thresh == 0.5):
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
        mAP_QbS = find_mAP_QbS(result, cf)
        mAP_QbE = find_mAP_QbE(result, cf)        
        print('Thresh val', my_thresh, '  mAP(QbS)=', mAP_QbS, '  ', 'mAP(QbE) = ', mAP_QbE, '----\n')


        
''' moment as a coeficent of variation'''
def word_str_moment(word_str):
    # word_str, loc = remove_single_words(word_str)
    vals, ids, idx = np.unique(word_str, return_index=True, return_inverse=True)
    vv = Counter(idx)    
    ss = np.zeros(None, dtype=float)   
    for i in range(0,  len(vv) ):
        ss = np.append(ss, vv[i]/len(vv) + (vv[i]/len(vv) )**2 + (vv[i]/len(vv) )**3 + (vv[i]/len(vv) )**4 )
    
    return np.std(ss)/np.mean(ss)


# http://www.nltk.org/howto/wordnet.html
def word_similarity_metric(list_of_words):
    ''' Example:   list_of_wrods  = ['hi', 'xerox', 'hi', 'xerox', 'dunk', 'hi']    
    print( word_similarity_metric(list_of_wrods) )
    
    '''
    
    list_of_words = list(list_of_words)
    
    ''''
    Test
    TEST
    
    
    list_of_words, loc = remove_single_words(list_of_words)  
    
    
    may be we should comment remove-single-words
    
    TODO
    '''
    
#    ss=0
#    list_len =  len(list_of_words)
#    for word in list_of_words:
#        list_of_words.pop(0)
#        ss += ListOfWords_to_ListOfWords_statistic(list([word]), list_of_words )
#    return 2*(ss/list_len)

    ss = np.zeros(None, dtype= float)    
    for word in list_of_words:
        list_of_words.pop(0)
        ss = np.append(ss, ListOfWords_to_ListOfWords_statistic(list([word]), list_of_words ) )
    return 2*np.std(ss)/np.mean(ss) # as a coeficent of variatio


def ListOfWords_to_ListOfWords_statistic(list1, list2):

# input args
# list1  ['choose', 'copy', 'define', 'copy', 'choose', 'choose']
# list2 has several words to be compared with the ones in lis1
    
    xx = np.zeros(len(list1) * len(list2), dtype=float)
       
    for word1 in list1:
        for word2 in list2:
            wordFromList1 = wordnet.synsets(word1)
            wordFromList2 = wordnet.synsets(word2)
            if wordFromList1 and wordFromList2: 
                s = wordFromList1[0].wup_similarity(wordFromList2[0])
                if s==None: s=0
                xx = np.append(xx, s)        
#     i=0 
#    for zz in xx:
#         xx[i] = int(0 if zz is None else zz)
#         i += 1
#        
#    return sum(xx)/( len(list1) * len(list2) )
#
    return xx



#def word_str_moment(word_str):
#    # word_str, loc = remove_single_words(word_str)
#    vals, ids, idx = np.unique(word_str, return_index=True, return_inverse=True)
#    vv = Counter(idx)
#    ss0 = sum(vv.values())
#    ss = 0    
#    for i in range(0,  len(vv) ):
#        ss += vv[i]/len(vv) + (vv[i]/len(vv) )**2 + (vv[i]/len(vv) )**3 + (vv[i]/len(vv) )**4
#    
#    ss= ss/ ss0
#    return ss
       
        
#def add_weights_of_words(data_set):
#    
#    word_str = []
#    for data, target, one_word_str, weights in data_set:
#        word_str.append(one_word_str)        
#    
#    # word_str=['apple','egg','apple','banana','egg','apple']
#    N = len(word_str)
#    wordfreq = [word_str.count(w) for w in word_str]
#    weights = 1 - np.array(wordfreq)/N
#    data_set.add_weights(weights)
#    return data_set



