import numpy as np
import logging
import sys

def build_phoc(words, split_character=None):
    '''  Calculate Pyramidal Histogram of Characters (PHOC) descriptor (see Almazan 2014).
    Args:
        words (str): words to calculate descriptor for                   
        split_character (str): special character to split the word strings into characters
        
    Returns:
        the PHOCs for the given words    '''
    
    # phoc_unigrams (str): string of all unigrams to use in the PHOC
    phoc_unigrams ='abcdefghijklmnopqrstuvwxyz0123456789ابجدهوزطحيكلمنسعفصقرشتثخذضظغ'    
    
    
    # unigram_levels (list of int): the levels for the unigrams in PHOC
    unigram_levels = [2,3,4,5]      
    
    # prepare output matrix
    logger = logging.getLogger('PHOCGenerator')   
    phoc_size = len(phoc_unigrams) * np.sum(unigram_levels)
    phocs = np.zeros((len(words), phoc_size))
    # prepare some lambda functions
    occupancy = lambda k, n: [float(k) / n, float(k + 1) / n]
    overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
    size = lambda region: region[1] - region[0]

    # map from character to alphabet position
    char_indices = {d: i for i, d in enumerate(phoc_unigrams)}

    # iterate through all the words
    for word_index, word in enumerate(words):
        if split_character is not None:
            word = word.split(split_character)
        n = len(word)
        for index, char in enumerate(word):
            char_occ = occupancy(index, n)
            if char not in char_indices:                                               
                logger.fatal('The unigram \'%s\' is unknown', char)
                sys.exit(0)                
            char_index = char_indices[char]
            for level in unigram_levels:
                for region in range(level):
                    region_occ = occupancy(region, level)
                    if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                        feat_vec_index = sum([l for l in unigram_levels if l < level]) * len(
                            phoc_unigrams) + region * len(phoc_unigrams) + char_index
                        phocs[word_index, feat_vec_index] = 1
       
    return phocs

# Testing the function
# words =['barcelona0', 'ali', 'uab', 'علي', 'اندريه']
# qry_phocs = build_phoc(words = words)
# print(qry_phocs)
# print('PHOCs has the size', np.shape(qry_phocs))


