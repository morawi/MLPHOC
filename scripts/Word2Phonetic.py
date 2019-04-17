#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:14:19 2019

@author: malrawi

http://www.spelltool.com/en/index.php
http://usefulenglish.ru/vocabulary/spelling-names-on-the-phone

"""


import chars2vec
import numpy as np
from scipy.ndimage import interpolation


# Load Inutition Engineering pretrained model
# Models names: 'eng_50', 'eng_100', 'eng_150'


dictionary  = {'a':'Alfa', 'b':'Bravo', 'c':'Charlie' , 'd':'Delta',
                'e'	:'Echo', 'f' :'Foxtrot', 'g' :'Golf', 'h':'Hotel', 'i':'India',
                'j'	:'Juliett', 'k'	:'Kilo', 'l':'Lima', 'm'	:'Mike',  'n':'November',
                'o'	:'Oscar', 'p':'Papa', 'q':'Quebec', 'r':'Romeo', 's'	:'Sierra',
                't'	:'Tango', 'u' :'Uniform', 'v':'Victor', 'w':'Whiskey', 'x':'Xray',
                'y'	:'Yankee', 'z':'Zulu',
                '0':'Nadazero', '1':'Unaone', '2':'Bissotwo', '3':'Terrathree',
                '4':'Kartefour', '5':'Pantafive', '6':'Soxisix', '7':'Setteseven',
                '8':'Oktoeight', '9':'Novenine', '-': 'splash'}

class Word2Phonetic():
    ''' Args-
    Models names: 'eng_50', 'eng_100', 'eng_150'   
    augment_rot_word: if True, the embbedding will be augmented by a rotated word
    this is useful to generated a sparce space of similar words; for example,
    silent and silence
    '''
    def __init__(self,  language_model = 'eng_50', max_word_len = 20): # phoc_vectors  have a size n_test_samplesXn_ensmbles
        self.c2v_model = chars2vec.load_model(language_model)
        self.len_vec =  len(self.c2v_model.vectorize_words(['dump']).squeeze())
        self.len_output = max_word_len*int(language_model.replace('eng_',''))
    
    def getvec(self, word, cf):  
        scale_factor = 2  
        word_embedding = np.empty(0, dtype = 'float32')        
        for char in word:
           word_embedding = np.append(word_embedding, self.get_phonetic(char, scale_factor) )
        if word_embedding.size==0:
            word_embedding = np.zeros(self.len_output, dtype = 'float32')                   
        else:
            zoom_factor = self.len_output / len(word_embedding)
            if zoom_factor>1:            
                word_embedding = interpolation.zoom(word_embedding, zoom_factor)
        
        return word_embedding
       
        
    def __getitem__(self, word): 
        
        return self.getvec(word, 0) # 0 is the cf value, neglected for now
        
    
    def get_phonetic(self, char, scale_factor):                   
                
        if char.isalnum():
            char_2word = dictionary[char]
            char_embedding = self.c2v_model.vectorize_words([char_2word]).squeeze()  # since the function accepts a list of wrods as input                                
            char_embedding = 0.5 + char_embedding /scale_factor # normalizing to posistive scale, dividing by 4 to keep it in the linear range of Sigmoid
        
        else: 
            char_embedding = np.empty(0, dtype = 'float32')             
#            char_embedding = self.get_padding(self.len_vec)             
                
        return char_embedding
               
#       
#y = Word2Phonetic()
#z = y['hi']