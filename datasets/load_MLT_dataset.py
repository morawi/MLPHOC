#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:07:33 2019

@author: malrawi
"""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter
import torch

def annotation_exists(word, cf):        
    char_indices = {d: i for i, d in enumerate(cf.phoc_unigrams)}    
    for index, char in enumerate(word):        
        if char not in char_indices:
            print(char,' ', end='')
            return False
    return True

class MLT_words(Dataset):
    def __init__(self, cf,  train=True, transform=None, data_idx = np.arange(1)):
        self.cf = cf
        self.train = train        
        self.transform = transform   
        self.img_name = []
        self.language = []
        self.word = []          
        # self.phoc_unigrams = self.collect_phoc_unigrams(phoc_languages = cf.MLT_languages)
        # self.phoc_levels = self.get_phoc_levels(phoc_languages = cf.MLT_languages)
        self.get_MLT_file_label()        
        self.no_word_per_language = Counter(self.language)
        len_data = len(self.img_name)
        if len(data_idx) == 1:  # this is safe as the lowest is one, when nothing is passed
            np.random.seed(cf.rnd_seed_value)
            self.data_idx = np.sort(np.random.choice(len_data, size=int(len_data * cf.split_MLT), 
                                                replace=False) )
        else:        
            all_idx = np.arange(0, len_data)
            self.data_idx = np.sort( np.setdiff1d(all_idx, data_idx, assume_unique=False) )
        
        self.img_name = [self.img_name[i] for i in self.data_idx]
        self.language = [self.language [i] for i in self.data_idx]
        self.word = [self.word[i] for i in self.data_idx]
         
        
        self.weights =  0 # depriciated for the time being; np.ones( len(self.file_label) , dtype = 'uint8' )               
    
    def get_phoc_levels(self, phoc_languages=['English','Arabic']):
        phocs = [ 
                  [2,3,4,5,6], 
                  [6,2,3,4,5],
                  [5,6,2,3,4],
                  [4,5,6,2,3],
                  [3,4,5,6,2],
                  [5,3,4,2,6],
                  [6,5,4,3,2],
                  [2,6,5,4,3], 
                  [3,2,6,5,4], 
                  [4,3,2,6,5]                      
                  ]
        phoc_levels = {}; 
        for ln in phoc_languages:
            #if ln in ['English', 'French',  'German', 'Italian']:
            if ln=='English' or ln=='French' or ln=='German' or ln=='Italian' : 
                phoc_levels['Latin']= phocs[0]
                phoc_languages.remove(ln)
        i=1     
        for ln in phoc_languages:
            phoc_levels[ln] =  phocs[i]
            i +=1
            
    
    def collect_phoc_unigrams(self, phoc_languages=['English','Arabic']):
        phoc_unigrams = {}
        phoc_unigrams['Latin']=''
        for ln in phoc_languages:
            phoc_unigrams[ln] = ''
        
            
        gt_tr = 'gt.txt'
        print('Collecting symbols from MLT dataset ........')
        with open(self.cf.dataset_path_MLT + gt_tr, 'r') as f_tr:
            data_tr = f_tr.readlines()
            for data_item in data_tr:  
                data_item = data_item.split(',') 
                if len(data_item) ==3: # some records have no language or error, like idx 2064; each field has exactly three enries, image_name, language, string code
                    im_nm, ln, ws  = data_item                    
                    if ln in phoc_languages: #if ln=='English' or ln=='Arabic': # pick only two languages, English and Arabic
                        ws = ws[:-1].lower() # taking out the new line symbol '\n'
                        if len(ws)<3: continue # removing small words or single characters
                        if ln=='English' or ln=='French' or ln=='German' or ln=='Italian' :                            
                            ln = 'Latin'                            
                        phoc_unigrams[ln] = ''.join(sorted( set(phoc_unigrams[ln] + ws) )) 
                        
        return phoc_unigrams                 
        
        
    def get_MLT_file_label(self):    
        Latin_langs = ['English', 'French','German','Italian']               
        gt_tr = 'gt.txt'
        print('Wrong char annotations will be printed out below........')
        with open(self.cf.dataset_path_MLT + gt_tr, 'r') as f_tr:
            data_tr = f_tr.readlines()
            for data_item in data_tr:  
                data_item = data_item.split(',') 
                if len(data_item) ==3: # some records have no language or error, like 2064
                    im_nm, ln, ws  = data_item                    
                    if ln in self.cf.MLT_languages: #if ln=='English' or ln=='Arabic': # pick only two languages, English and Arabic
                        ws = ws[:-1].lower() # taking out the new line symbol '\n'
                        if len(ws)<3: continue # removing small words or single characters
                        if not annotation_exists(ws, self.cf): continue  
                        if ln in Latin_langs and self.cf.MLT_latin_script_vs_others == True:
                            ln=='Latin'
                        self.img_name.append(im_nm)
                        self.language.append(ln)
                        self.word.append(ws)
                                                     
    
    def __getitem__(self, index):            
        img = Image.open(self.cf.dataset_path_MLT + 'MLT_images/' + self.img_name[index])        
        word_str = self.word[index]
        if not(self.cf.H_MLT_scale == 0): # resizing just the height             
            new_w = int(img.size[0]*self.cf.H_MLT_scale/img.size[1])
            if new_w>self.cf.MAX_IMAGE_WIDTH: 
                new_w = self.cf.MAX_IMAGE_WIDTH
            img = img.resize( (new_w, self.cf.H_MLT_scale), Image.ANTIALIAS)               
        
        # target = self.cf.PHOC(word_str, cf = self.cf)    
        if self.cf.task_type=='script_identification':            
            target = torch.from_numpy(self.cf.PHOC(self.language[index].lower()+ 
                                  self.cf.language_hash_code[self.language[index]], self.cf) ) # language_name + hashcode
        else:            
            target = torch.from_numpy( self.cf.PHOC(word_str, cf = self.cf, mode= 'printed-writing')   )
                
        if img.mode !='RGB':
            img = img.convert('RGB')
                    
        if self.transform:            
            img = self.transform(img)    
        
        return img, target, word_str, self.weights

    def __len__(self):
        return len(self.word)
     
    def num_classes(self):
        if self.cf.task_type=='script_identification':
            x= self.cf.PHOC(self.language[0].lower()+ 
                                  self.hash_code[self.language[0]], self.cf) # 
            return len(x)
        else:
            return len(self.cf.PHOC('dump', self.cf)) # pasing 'dump' word to get the length


