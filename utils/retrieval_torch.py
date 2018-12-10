#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:53:18 2018

@author: malrawi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Jul 10, 2016
@author: ssudholt
https://github.com/ssudholt/phocnet
'''
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import torch

from sklearn.metrics.pairwise import cosine_similarity


# from module import Module
# from .. import functional as F

def map_from_feature_matrix(features, labels, metric, drop_first):
    '''
    Computes mAP and APs from a given matrix of feature vectors
    Each sample is used as a query once and all the other samples are
    used for testing. The user can specify whether he wants to include
    the query in the test results as well or not.
    
    Args:
        features (2d-ndarray): the feature representation from which to compute the mAP
        labels (1d-ndarray or list): the labels corresponding to the features (either numeric or characters)
        metric (string): the metric to be used in calculating the mAP
        drop_first (bool): whether to drop the first retrieval result or not
        
        metric types : str. The distance function can be ‘braycurtis’,  ‘canberra’, ‘chebyshev’, 
         ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, 
         ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, 
         ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’.
        
    '''
    # argument error checks
    if features.shape[0] != len(labels):
        raise ValueError('The number of feature vectors and number of labels must match')
    # compute the pairwise distances from the
    # features
#    dists = pdist(X=features, metric=metric)
#    dists = squareform(dists)  # using torch based, batch pased, cosine cosine_similarity_n_space instead
    
    dists = cosine_similarity_n_space(m1 = features, m2=None, dist_batch_size=10000)
    
    inds = np.argsort(dists, axis=1)
    del dists # to free memory
    retr_mat = np.tile(labels, (features.shape[0],1))
    
    # compute two matrices for selecting rows and columns
    # from the label matrix
    # -> advanced indexing
    row_selector = np.transpose(np.tile(np.arange(features.shape[0]), (features.shape[0],1)))
    retr_mat = retr_mat[row_selector, inds]
    
    # create the relevance matrix
    rel_matrix = retr_mat == np.atleast_2d(labels).T
    if drop_first:
        rel_matrix = rel_matrix[:,1:]
        
    # calculate mAP and APs
    map_calc = MeanAveragePrecision()
    avg_precs = np.array([map_calc.average_precision(row) for row in rel_matrix])
    mAP = np.mean(avg_precs)
    return mAP, avg_precs

def map_from_query_test_feature_matrices(query_features, test_features, query_labels, test_labels, 
                                         metric, drop_first=False):
    '''
    Computes mAP and APs for a given matrix of query representations
    and another matrix of test representations
    Each query is used once to rank the test samples.
    
    Args:
        query_features (2d-ndarray): the feature representation for the queries
        query_labels (1d-ndarray or list): the labels corresponding to the queries (either numeric or characters)
        test_features (2d-ndarray): the feature representation for the test samples
        test_labels (m1, m21d-ndarray or list): the labels corresponding to the test samples (either numeric or characters)
        metric (string): the metric to be used in calculating the mAP, which is a metric used by cdist of
        scipy.spatial.distance. See the forms below.
        drop_first (bool): whether to drop the first retrieval result or not
        
         metric types : str. The distance function can be ‘braycurtis’,  ‘canberra’, ‘chebyshev’, 
         ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, 
         ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, 
         ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’.
        
    '''
    # some argument error checking
    if query_features.shape[1] != test_features.shape[1]:
        raise ValueError('Shape mismatch')
    if query_features.shape[0] != len(query_labels):
        raise ValueError('The number of query feature vectors and query labels does not match')
    if test_features.shape[0] != len(test_labels):
        raise ValueError('The number of test feature vectors and test labels does not match')
    
    # compute the nearest neighbors
    # dist_mat = cdist(XA=query_features, XB=test_features, metric=metric)
    dist_mat = cosine_similarity_n_space(query_features, test_features, dist_batch_size=10000)
    retrieval_indices = np.argsort(dist_mat, axis=1)
    del dist_mat # to free memory
    
    # create the retrieval matrix
    retr_mat = np.tile(test_labels, (len(query_labels),1))
    row_selector = np.transpose(np.tile(np.arange(len(query_labels)), (len(test_labels),1)))
    retr_mat = retr_mat[row_selector, retrieval_indices]
    
    # create the relevance matrix
    relevance_matrix = retr_mat == np.atleast_2d(query_labels).T
    if drop_first:
        relevance_matrix = relevance_matrix[:,1:]
    
    # calculate mAP and APs
    mapCalc = MeanAveragePrecision()
    avg_precs = np.array([mapCalc.average_precision(row) for row in relevance_matrix], ndmin=2).flatten()
    mAP = np.mean(avg_precs)    
    return mAP, avg_precs

class IterativeMean(object):
    '''
    Class for iteratively computing a mean. With every new value (@see: _add_value)
    the mean will be updated 
    '''
    
    def __init__(self, mean_init=0.0):
        self.__mean = mean_init
        self.__N = 0.0
        
    def add_value(self, value):
        '''
        Updates the mean with respect to value
        
        Args:
            value (float): The value that will be incorporated in the mean
        '''  
        self.__mean = (self.__N / (self.__N + 1)) * self.__mean + (1.0 / (self.__N + 1)) * value
        self.__N += 1
        
    def get_mean(self):
        return self.__mean
    
    def reset(self):
        self.__mean = 0.0
        self.__N = 0.0
         

class MeanAveragePrecision(IterativeMean):
    '''
    Computes average precision values and iteratively updates their mean
    '''
    def __init__(self):
        super(MeanAveragePrecision, self).__init__()        
        
    def average_precision(self, ret_vec_relevance, gt_relevance_num=None):
        '''
        Computes the average precision and updates the mean average precision
        
        Args:
            ret_vec_relevance (1d-ndarray): array containing ground truth (gt) relevance values
            gt_relevance_num (int): The number of relevant samples in retrieval. If None the sum
                                    over the retrieval gt list is used.
        '''
        ret_vec_cumsum = np.cumsum(ret_vec_relevance, dtype=float)
        ret_vec_range = np.arange(1, ret_vec_relevance.size + 1)
        ret_vec_precision = ret_vec_cumsum / ret_vec_range
        
        if gt_relevance_num is None:
            n_relevance = ret_vec_relevance.sum()
        else:
            n_relevance = gt_relevance_num

        if n_relevance > 0:
            ret_vec_ap = (ret_vec_precision * ret_vec_relevance).sum() / n_relevance
        else:
            ret_vec_ap = 0.0
        
        super(MeanAveragePrecision, self).add_value(ret_vec_ap)
        
        return ret_vec_ap


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def cosine_similarity_n_space(m1, m2, dist_batch_size=100):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    m2 = m1 if m2 is None else m2
    assert m1.shape[1] == m2.shape[1]
    
    result = torch.zeros([1, m2.shape[0]])
    
    for row_i in range(0, int(m1.shape[0] / dist_batch_size) + 1):
        start = row_i * dist_batch_size
        end = min([(row_i + 1) * dist_batch_size, m1.shape[0]])
        if end <= start:
            break # cause I'm too lazy to elegantly handle edge cases
        rows = m1[start: end] 
        # sim = cosine_similarity(rows, m2) # rows is O(1) size
        
        sim = cosine_distance_torch(torch.tensor(rows).to(device), 
                                    torch.tensor(m2).to(device))
        
        result = torch.cat( (result, sim.cpu()), 0)
        
               
    result = result[1:, :] # deleting the first row, as it was used for setting the size only
    del sim
    return result.numpy() # return 1 - ret # should be used with sklearn cosine_similarity




#input1 = torch.randn(5, 7)
#input2 = torch.randn(5, 7)
#
#
#dist_mat = cdist(input1.numpy(), input2.numpy(), metric='cosine')
#print(dist_mat)
##
## xx= cosine_distance_torch(input1, input2, 2.4)
## print(xx)
#yy = cosine_similarity_n_space(input1.numpy(), input2.numpy(), dist_batch_size=10000)
#print(yy)
#
## https://stackoverflow.com/questions/40900608/cosine-similarity-on-large-sparse-matrix-with-numpy

