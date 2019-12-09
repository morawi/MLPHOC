#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:19:41 2019

@author: malrawi

predict labels from a multi-hot binary vector input

"""



import numpy as np
from collections import Counter
from utils.torch_cosine import cosine_similarity_n_space as TorchCosine
# from torch_cosine import cosine_similarity_n_space as TorchCosine
from collections import OrderedDict

''' to compare with other distances from sklearn and scipy'''




def predict_labels(y_true, y_pred, true_labels):   
    ''' 
    The function mainly uses torch.cosine distance to find the predicted labesl
    from a model output. The function is based on finding the distance between
    the prediction vectors and the ground truth unique vectors and their 
    corresponding labels. The function assuems that each prediction and
    the ground truth vectros are represented in multi-hot value, or even real 
    values. 
    Other distances based  on sklearn and scipy str also supported, after commenting the cosine distance
    and uncommenting the sklearn or scipy distnace, based on Eucleadine, but
    can be easily changed to any other pair-wise distance. 
     
    # sklearn and scipy distances can also be used
    from sklearn.metrics.pairwise import euclidean_distances as dist
    from scipy.spatial import distance 

    dist_mat = distance.cdist(y_true, y_pred, "Euclidean")
    dist_mat = dist(y_true, y_pred)           
    
    Args in --           
    y_pred: mxn sized matrix, m is the number of samples, each sample is a 
    vector of size n. Each sample could be binary (one hot, or even mutli-hot), 
    or even continuous real-valued vector. y_pred is the output from the
    classifiction model.
    y_true: lxn size matrix, l is the number of samples, each sample is vector
    of size n. 
    true_labels: n sized list, labels corresponding to y_true 
    If the ground_truth and the corresponding true_labels are not unique, this function will take care 
    of this problem.
        
        
        '''        
    d = OrderedDict((x, true_labels.index(x)) for x in true_labels)
    
    unique_labels = list( d.keys())
    unique__label_idx = list(d.values() )
    y_true= y_true[unique__label_idx, :]
    dist_mat = TorchCosine(y_true, y_pred)
    dist_indices = np.argmin(abs(dist_mat), axis=0)  # finds the idx of min vals      
    predicte_labels = [unique_labels[i] for i in dist_indices]
#
    return predicte_labels
        
    

#'''' Example  '''
#        
#gt_mat = np.array([
#              [0,0,0],  # A
#              [0,0,1],  # B
#              [0,1,0],  # C
#              [0,1,0],  # C
#              [1,0,0], #  D 
#              [1,0,1],  # E
#              [1,1,0]   # F
#                   ])
#''' Note that ground truth should not necessarily contain unique samples,
#    but i've done it this way for illustrative purpose. It is possible that
#    the ground truth has repetitions of labels, as the same label might be 
#    associated to the same input image, speech, text, i.e. input sample '''
#    
#gt_labels = ["A", "B", "C", "C", "D", "E", "F"  ]    
#
#pred_mat = np.array([
#              [1,1,0],  
#              [1,0,0],
#              [1,1,0],
#              [1,1,0],
#              [1,1,0], 
#              [0,0,1],
#              [0,0,1] ])
#
#
#x = predict_labels(gt_mat, pred_mat, gt_labels)
#print(x)
