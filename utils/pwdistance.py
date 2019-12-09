#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:36:33 2019

@author: malrawi
"""

import numpy as np
from collections import Counter

from utils.torch_cosine import cosine_similarity_n_space as TorchCosine

''' to compare with other distances from sklearn and scipy'''
from sklearn.metrics import accuracy_score as acc_sklearn
from sklearn.metrics.pairwise import euclidean_distances as dist
# from scipy.spatial import distance 



def accuracy_score(y_true, y_pred, labels_true, transform_labels=None, 
                   normalize = True, diagnostics=False, verbose = False):   
    ''' 
    The function mainly uses torch.cosine distance. Other distances based 
    on sklearn and scipy str also supported, after commenting the cosine distance
    and uncommenting the sklearn or scipy distnace, based on Eucleadine, but
    can be easily changed to any other pair-wise distance. 
    
    Args in --    
        
    y_true: ground truth matrix as numpy array of size n_samplesXfeature_size.
        
    y_pred: predicted matrix as numpy array of size n_samplesXfeature_size
    we choose to use similar notations of those in sklearn and scipy.
    
    labels_true: These are the ground truth labels associated to y_true, that is
    each sample (row) in y_true has a corresponding label denoted as a string,
    could be the name of a class category.
    
    transform_labels: map labels to anothre class, mapping from fine to
    coarse classes, and vice versa.
    
    diagnostics: correctly and incorrectly identifid labels,
    and the prediction accuracy by matching y_true and y_pred.
    
    verbose: prints the predicted labels and their indices in y_pred by 
    matching y_true and y_pred.
    
    normalize: has similar effect of sklearn metrics, that is, if False, then
    it will not divide by the number of samples, which is len(labels_true) 
    or y_true.size[0]    
    
    
    Outputs --
    accuracy: Accuracy classification score. In multilabel classification, this 
    function computes subset accuracy: the set of labels predicted for a sample 
    that has the minimum distance with the an entry in y_true.  
    cnt_corr: a Counter object detailing the times each label has been 
    correctly identified
    cnt_incorr: a Counter object detailing the times each label has been 
    incorrectly identified
    predicted_labels: the predicted labels, one can use these, for instnace, 
    to construct the confusion matrix
    
    
        
        '''
    
    transform_labels(labels_true) if transform_labels!=None else None
    cnt_corr = Counter(); # counter to hold the correctly identified samples
    cnt_incorr = Counter();  # counter to hold the incorrectly identified samples
    predicted_labels = [] # 
#   dist_mat = distance.cdist(y_true, y_pred, "Euclidean")
#    dist_mat = dist(y_pred, y_true)           
    dist_mat = TorchCosine(y_true, y_pred)
    dist_indices = np.argmin(abs(dist_mat), axis=0)  # finds the idx of min vals      
        
    for j, label_idx in enumerate(dist_indices):
        predicted_labels.append(labels_true[label_idx]) # I'd like to store these, although we can skip this step bu using labels_tru[i] below         
        if predicted_labels[j] == labels_true[j]: 
            cnt_corr[labels_true[label_idx]] += 1 # the correctly predicted for each label
        else:
            cnt_incorr[labels_true[label_idx]] +=1 # the incorrectly predicted for each label
        
    accuracy =  sum( cnt_corr.values()) / len(labels_true) if normalize else sum( cnt_corr.values())
    if diagnostics:
        if verbose:
            print(dist_mat)
            print('ground truth labels:', labels_true)            
            print('predicted labels:   ', predicted_labels)        
        
        print('indices of predicted labels:', dist_indices)
        print('correctly identified labels:   ', cnt_corr)
        print('in-correctly identified labels:', cnt_incorr)
        print('total number of correct samples:', sum( cnt_corr.values()), 'out of', len(labels_true), '; accuracy is:', accuracy )
                
    
    return accuracy, cnt_corr, cnt_incorr, predicted_labels
        
    

''' 
Example of usage and explanation of the solution we are providing. 
Below we have a matrix with 7 ground truth vectors and seven ground truth labels. 
Each vectro/feature has 3 dimensions

After performing prediction using some prediction 'model', one gets the output, 
name it pred_mat, that also has seven vectors. 
The hypothesis is that the predicted labels (pred_mat) should match the ground truth (gt_mat). 
But, how can one get the prediction labels fro pred_mat? 
We shall use a pair-wise distance metric. For more info see:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

The problem we are therfore solving is:
- comparing the prediction to the ground truth using a distance metric, and then 
getting the indices of the min distance values;
- obtaining the predicted labels by projecting the indices on the ground truth label; and
- estimating then the accuracy by finding the match between the predicted labels and 
the ground truth labels.

Explanation of the run:
So, after calculating the distance, the predicted labels will be 
 ['G', 'E', 'G', 'D', 'D', 'D', 'D']
Now, comparing gt_labels and predicted labels shows that there is only on correct match
which is D (fourth sample in pred_mat)

Additional note: Using gt_mapping
   
gt_mapped = the_mapping_function(gt_labels) 
    It is possible to map the ground truth is mapped to another space, 
    for example, words to a specific language, or, to another higher level class,
    Another example is coarse to fine-grained classification; i.e., 
    major classes with sub class categories. For example, class "B"(car) and 
    "C"(bus) are mapped to vehicles, and "A" to animals.
    the_mapping_function() can be added by the user   


'''
        
#gt_mat = np.array([
#              [0,0,0], 
#              [0,0,1],
#              [0,1,0],
#              [0,1,1],
#              [1,0,0], 
#              [1,0,1],
#              [1,1,0]
#                   ])
#''' Note that ground truth should not necessarily contain unique samples,
#    but i've done it this way for illustrative purpose. It is possible that
#    the ground truth has repetitions of labels, as the same label might be 
#    associated to the same input image, speech, text, i.e. input sample '''
#    
#gt_labels = ["A", "B", "C", "D", "E", "F", "G"  ]    
#
#pred_mat = np.array([
#              [1,1,0],  
#              [1,0,0],
#              [1,1,0],
#              [1,1,1],
#              [1,1,1], 
#              [1,1,1],
#              [1,1,1] ])
#
#
#accuracy_score(gt_mat, pred_mat, gt_labels, diagnostics=True, verbose=False)
#print('sklearn accuracy_score gave:', acc_sklearn(gt_mat, pred_mat))
#
#


