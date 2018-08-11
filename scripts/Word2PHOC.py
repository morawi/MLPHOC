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

    # prepare output matrix
    phoc_size = len(cf.phoc_unigrams) * np.sum(cf.unigram_levels)
    phoc = np.zeros(phoc_size,  dtype='float32')

    # prepare some lambda functions
    occupancy = lambda k, n: [float(k) / n, float(k + 1) / n]
    overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
    size = lambda region: region[1] - region[0]

    # map from character to alphabet position
    char_indices = {d: i for i, d in enumerate(cf.phoc_unigrams)}

    n = len(word)
    for index, char in enumerate(word):
        char_occ = occupancy(index, n)
        if char not in char_indices:
            if  cf.keep_non_alphabet_of_GW_in_analysis==False and cf.dataset_name=='WG':
                continue
            else:    
                logger.fatal('The unigram \'%s\' is unknown', char)
                sys.exit(0)

        char_index = char_indices[char]
        for level in cf.unigram_levels:
            for region in range(level):
                region_occ = occupancy(region, level)
                if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                    feat_vec_index = sum([l for l in cf.unigram_levels if l < level]) * len(
                        cf.phoc_unigrams) + region * len(cf.phoc_unigrams) + char_index
                    phoc[feat_vec_index] = 1
       
    if cf.phoc_tolerance>0:
        phoc = np.array([1-cf.tolerance if x>0.5 else cf.tolerance for x in phoc]) #, dtype = 'float32')
    return phoc 

