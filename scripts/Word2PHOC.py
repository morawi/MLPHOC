import numpy as np
import logging
import sys

""" adapted from @author: ssudholt, 
 https://github.com/ssudholt/phocnet/blob/master/src/phocnet/attributes/phoc.py
 
 - Mohammed Al-Rawi
 """

def build_phoc(word, cf): # alphabet = 'multiple', unigram_levels = [2,3,4,5]):
    '''  Calculates Pyramidal Histogram of Characters (PHOC) descriptor (see Almazan 2014).
    Args:
        words (str): word to calculate descriptor for
        cf.alphabet (str): choose the alphabet to compute the PHOC
        cf.unigram_levels (array): [2,3,4,5]
        
    Returns:
        the PHOCs for the given words    '''

    logger = logging.getLogger('PHOCGenerator')
    # phoc_unigrams (str): string of all unigrams to use in the PHOC
    
    if cf.dataset_name == 'WG':
        phoc_unigrams =".0123456789abcdefghijklmnopqrstuvwxyz.,-;':()£|"
        # phoc_unigrams ='abcdefghijklmnopqrstuvwxyz0123456789'
    elif cf.dataset_name =='IFN':
        phoc_unigrams ="0123456789أءابجدهوزطحيكلمنسعفصقرشتثخذضظغةى.ئإآ\'ّ''"
    elif cf.dataset_name == 'WG+IFN':
        if cf.keep_non_alphabet_in_GW==True:
            phoc_unigrams ="abcdefghijklmnopqrstuvwxyz,-;':()£|0123456789أءابجدهوزطحيكلمنسعفصقرشتثخذضظغةى.ئإآ\'ّ''"
        else:
            phoc_unigrams ="abcdefghijklmnopqrstuvwxyz0123456789أءابجدهوزطحيكلمنسعفصقرشتثخذضظغةى.ئإآ\'ّ''"
            
    elif cf.dataset_name == 'IAM':
        x = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        phoc_unigrams = ''.join(map(str, x))
    else: 
        logger.fatal("Datasets to use: 'WG', 'IFN', 'IAM', or 'WG+IAM' ")
        sys.exit(0)
         

    # prepare output matrix
    phoc_size = len(phoc_unigrams) * np.sum(cf.unigram_levels)
    phoc = np.zeros(phoc_size)

    # prepare some lambda functions
    occupancy = lambda k, n: [float(k) / n, float(k + 1) / n]
    overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
    size = lambda region: region[1] - region[0]

    # map from character to alphabet position
    char_indices = {d: i for i, d in enumerate(phoc_unigrams)}

    n = len(word)
    for index, char in enumerate(word):
        char_occ = occupancy(index, n)
        if char not in char_indices:
            logger.fatal('The unigram \'%s\' is unknown', char)
             #print(' ', char, end="" )
            sys.exit(0)

        char_index = char_indices[char]
        for level in cf.unigram_levels:
            for region in range(level):
                region_occ = occupancy(region, level)
                if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                    feat_vec_index = sum([l for l in cf.unigram_levels if l < level]) * len(
                        phoc_unigrams) + region * len(phoc_unigrams) + char_index
                    phoc[feat_vec_index] = 1
       
    return phoc


