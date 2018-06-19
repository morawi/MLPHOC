import numpy as np
import logging
import sys

def build_phoc(word, alphabet='multiple', unigram_levels = [2,3,4,5]):
    '''  Calculate Pyramidal Histogram of Characters (PHOC) descriptor (see Almazan 2014).
    Args:
        words (str): word to calculate descriptor for
        alphabet (str): choose the alphabet to compute the PHOC
        unigram_levels (array): [2,3,4,5]
        
    Returns:
        the PHOCs for the given words    '''

    logger = logging.getLogger('PHOCGenerator')
    # phoc_unigrams (str): string of all unigrams to use in the PHOC
    if alphabet == 'english':
        phoc_unigrams ='abcdefghijklmnopqrstuvwxyz0123456789'
    elif alphabet == 'arabic':
        phoc_unigrams ='0123456789أءابجدهوزطحيكلمنسعفصقرشتثخذضظغةى.ئإآ\'ّ'''
    elif alphabet == 'multiple':
        phoc_unigrams ='abcdefghijklmnopqrstuvwxyz0123456789أءابجدهوزطحيكلمنسعفصقرشتثخذضظغةى.ئإآ\'ّ'''
    else:
        logger.fatal('The alphabet flag (str) should be: english, arabic or multiple')
        sys.exit(0)
    
    # unigram_levels (list of int): the levels for the unigrams in PHOC
    unigram_levels = unigram_levels
    # prepare output matrix
    phoc_size = len(phoc_unigrams) * np.sum(unigram_levels)
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
        for level in unigram_levels:
            for region in range(level):
                region_occ = occupancy(region, level)
                if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                    feat_vec_index = sum([l for l in unigram_levels if l < level]) * len(
                        phoc_unigrams) + region * len(phoc_unigrams) + char_index
                    phoc[feat_vec_index] = 1
       
    return phoc


