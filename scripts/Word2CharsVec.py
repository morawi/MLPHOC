#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:57:02 2019

@author: malrawi

using: https://github.com/IntuitionEngineeringTeam/chars2vec
https://hackernoon.com/chars2vec-character-based-language-model-for-handling-real-world-texts-with-spelling-errors-and-a3e4053a147d
"""


import chars2vec
import numpy as np

# import time


# Load Inutition Engineering pretrained model
# Models names: 'eng_50', 'eng_100', 'eng_150'


class Chars2Vec():
    ''' Args-
    Models names: 'eng_50', 'eng_100', 'eng_150'   
    augment_rot_word: if True, the embbedding will be augmented by a rotated word
    this is useful to generated a sparce space of similar words; for example,
    silent and silence
    '''
    def __init__(self, language_model = 'eng_50', augment_rot_word = False): # phoc_vectors  have a size n_test_samplesXn_ensmbles
        self.c2v_model = chars2vec.load_model(language_model)  
        self.augment_rot_word= augment_rot_word
    
    def __getitem__(self, word):        
        return self.getvec(word, 0)
    
    def rotate_string(self, strg, n):
        return strg[n:] + strg[:n]

        
    def getvec(self, word, cf):    # cf has to be passed to cope with other functions' forms, like Word2PHOC
        
        scale_factor = 2
        
        word_embedding = self.c2v_model.vectorize_words([word]).squeeze()  # since the function accepts a list of wrods as input          
        
        if self.augment_rot_word:
            word_rotated  = self.rotate_string(word, len(word)//2)
            word_embedding = np.append(word_embedding,  self.c2v_model.vectorize_words([word_rotated]).squeeze() )
            
        word_embedding = .5 + word_embedding /scale_factor # normalizing to posistive scale, dividing by 4 to keep it in the linear range of Sigmoid
        return word_embedding  # scaling to keep the range within the output o the activagtion function
        
    
#x = Chars2Vec(augment_rot_word=True)
#z= x['hello']
#


##start_timer = time.time(); 
##for i in range(len(test_loader)):
##   test_loader.dataset[i]; 
##s = time.time() - start_timer; print(s)
#
## First, you're going to need to import wordnet: 
#


#from nltk.corpus import wordnet 
## Then, we're going to use the term "program" to find synsets like so: 
#syns = wordnet.synsets("program") 
#
## An example of a synset: 
#print(syns[0].name()) 
#
## Just the word: 
#print(syns[0].lemmas()[0].name()) 
#
## Definition of that first synset: 
#print(syns[0].definition()) 
#
## Examples of the word in use in sentences: 
#print(syns[0].examples()) 
