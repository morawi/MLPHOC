#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:45:02 2018

@author: malrawi
"""

'''
Sentence cleaning and preprocessing
https://github.com/PetrochukM/PyTorch-NLP/blob/master/torchnlp/word_to_vector/glove.py
https://hackernoon.com/word2vec-part-1-fe2ec6514d70
https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.word_to_vector.html

https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy

https://fasttext.cc/docs/en/english-vectors.html
https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c

Google trained Word2Vec model:
http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/

https://machinelearningmastery.com/prepare-movie-review-data-sentiment-analysis/

The data has been used for a few related natural language processing tasks. For classification, the performance of classical models (such as Support Vector Machines) on the data is in the range of high 70% to low 80% (e.g. 78%-to-82%).
More sophisticated data preparation may see results as high as 86% with 10-fold cross validation. This gives us a ballpark of low-to-mid 80s if we were looking to use this dataset in experiments on modern methods.

Datasets:
https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.datasets.html
The original paper and dataset
http://ai.stanford.edu/~amaas/data/sentiment/

https://machinelearningmastery.com/develop-word-embeddings-python-gensim/



Using NLTK’s Wordnet to find the meanings of words, synonyms, antonyms, and more. In addition, we use WordNetLemmatizer to get the root word.
https://datascienceplus.com/topic-modeling-in-python-with-nltk-and-gensim/

Building your own model, based on your sentences
from gensim.models import Word2Vec
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['sentence'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)

'''
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchnlp.datasets import imdb_dataset
import re 
import os
import numpy as np
from nltk.corpus import brown
import gensim
from functools import reduce
from torchnlp.word_to_vector import FastText
from torchnlp.word_to_vector import GloVe
from torchnlp.word_to_vector import CharNGram
from string import punctuation
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import torch
# from nltk import word_tokenize, pos_tag
# from gensim.summarization.textcleaner import split_sentences



#text = word_tokenize("They refuse to permit us to obtain the refuse permit")
#xx = pos_tag(text)
#[('They', 'PRP'), ('refuse', 'VBP'), ('to', 'TO'), ('permit', 'VB'), ('us', 'PRP'),
#('to', 'TO'), ('obtain', 'VB'), ('the', 'DT'), ('refuse', 'NN'), ('permit', 'NN')]



''' Either this text cleaning '''
def clean_text_1(text):
    stop_words = set(stopwords.words("english")) 
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

''' or, this text  clearning '''
def clean_text_2(text, stop_words):      
    text=text.lower() 
    text = re.sub("\>|\<|br|/|\x85|\\\|r'http\S+'|\"|\.|\,|\;|\:|\(|\)", '', text)    
    text = re.sub("\#", 'number ', text) 
    repls = ('?', ' questionmark'), ('!', ' exclamation_mark'),
    ("'re", ' are'), ("it's", 'it is'), ("n't", ' not') ,  (':', ' '), ('#', ' number ')          
    text  = reduce(lambda a, kv: a.replace(*kv), repls, text)
    text = text.replace('-', ' ')      
    text = text.replace("'s", ' ')   
    text = text.replace("'", ' ')    
    text = text.split()	# split into tokens by white space    
    text = [w for w in text if not w in stop_words]         
    table = str.maketrans('', '', punctuation) # remove punctuation from each token
    text = [w.translate(table) for w in text]	
    text = [word for word in text if word.isalpha()] # remove remaining tokens that are not alphabetic	
    text = [word for word in text if len(word) > 1] # filter out short tokens
    return text
        
    
def build_w2v_model(cf, reviews):
      
    vocab = Counter() # define vocab
    for review in reviews:
         vocab.update(review['text'])   # print(vocab.most_common(50)) # print the top words in the vocab        
    tokens = [k for k, c in vocab.items() if c >= cf.imdb_min_occurane] # keep tokens with > 5 occurrence    
    model = gensim.models.Word2Vec([tokens], min_count=1, size = 300)    
    return model

def get_word_vec_model(cf, reviews):
    print('--- Loading', cf.word_corpus_4_text_understanding, " pre-trained Word2Vec model")
    if  cf.word_corpus_4_text_understanding == 'Google_news' :        
        word_vec_model = gensim.models.KeyedVectors.load_word2vec_format(cf.folder_of_data + 
                                                                    '/models/GoogleNews-vectors-negative300.bin', binary=True)      
        
    elif cf.word_corpus_4_text_understanding=='Brown':        
        word_vec_model = gensim.models.Word2Vec(brown.sents()) # using Brown corpus
    elif cf.word_corpus_4_text_understanding == 'Fasttext':
        word_vec_model = FastText(language="simple")
    elif cf.word_corpus_4_text_understanding == 'Glove':
        word_vec_model = GloVe(name='6B', dim=300) 
    elif cf.word_corpus_4_text_understanding == 'CharNGram':
        word_vec_model = CharNGram()
    elif cf.word_corpus_4_text_understanding == 'Custom':
        word_vec_model = build_w2v_model(cf, reviews)        
    else:
        print('Error: Please select a word vector model')
    print('--- Loading', cf.word_corpus_4_text_understanding, ' done')
    
    return word_vec_model # , custom_model


class IMDB_dataset(Dataset):
    
    # w2v dimension is 300
    google_model   = None    
    fasttext_model = None
    glove_model    = None
    custom_model   = None
    # brown_model = None # w2v dimension is 100 here, not compatiable
    # CharNGram_model = None # w2v dimension is 100 here, not compatiable
        
    def __init__(self, cf, mode='train', transform = None):
        # mode: 'train' or 'test'            
        self.cf = cf
        self.mode = mode        
        self.transform = transform      
        if mode=='train': # For some reason, this has to be done this way..train = True, then, test=True!
            self.data = imdb_dataset(directory='/home/malrawi/Desktop/My Programs/all_data/imdb/', train = True)
            self.clean_all_text() # the clean text will replace the original one in self.data, it is necessary to do it here, as it the text might be used to build the w2v models        
            self.load_w2v_models()
        
        else:
            self.data = imdb_dataset(directory='/home/malrawi/Desktop/My Programs/all_data/imdb/', test=True)
            self.clean_all_text()                     
    
    def load_w2v_models(self):
        original_path =  os.getcwd()
        os.chdir(self.cf.folder_of_data + 'all_data/') # this is where the data models are stored
        
        print('loading google w2v')
        IMDB_dataset.google_model = gensim.models.KeyedVectors.load_word2vec_format(self.cf.folder_of_data + 
                                                                '/models/GoogleNews-vectors-negative300.bin', binary=True)      
        print('loading fasttext w2v')
        IMDB_dataset.fasttext_model = FastText(language="simple")
        print('loading glove w2v')
        IMDB_dataset.glove_model = GloVe(name='6B', dim=300)
        print('building/loading custom w2v')
        IMDB_dataset.custom_model = build_w2v_model(self.cf, self.data)   
        
        os.chdir(original_path) # restore the original path
        
    def clean_all_text(self):
        print('.....................Cleaning all IMDB text....................')
        stop_words = set(stopwords.words('english')) # filter out stop words
        i=0
        for review in self.data:
            self.data[i]['text'] = clean_text_2(review['text'], stop_words)
            i=i+1      
        
    def normalized_w2v(self, vec):
        min_v = min(vec)
        vec = (vec-min_v)/(max(vec) - min_v + 0.00000287361) - 1 # the tolerance value 0.00000287361 to prevent zero division if max is 0
        return vec
    
    def text_to_image_2(self, text):        
        img  = np.zeros([ self.cf.MAX_IMAGE_HEIGHT, self.cf.W_imdb_width], dtype='float32')   
        one_w2v = True # temporarly trying one w2v model
        if one_w2v:
            w2v = np.array(np.column_stack([IMDB_dataset.glove_model[w] for w in text])) # This model has no vocabulary        
        else:            
            w2v_google = np.array(np.column_stack(
                    [self.normalized_w2v(IMDB_dataset.google_model[w]) for w in text if w in IMDB_dataset.google_model.wv]))        
            if w2v_google.shape[1]<self.cf.W_imdb_width:
                w2v_fastext = np.array(np.column_stack([self.normalized_w2v(IMDB_dataset.fasttext_model[w]) for w in text])) # This model has no vocabulary
                w2v = np.concatenate((w2v_google, w2v_fastext), axis=1)
                if w2v.shape[1]<self.cf.W_imdb_width:
                    w2v_glove = np.array(np.column_stack([self.normalized_w2v(IMDB_dataset.glove_model[w]) for w in text])) # This model has no vocabulary        
                    w2v = np.concatenate((w2v, w2v_glove), axis=1)
                    if w2v.shape[1]<self.cf.W_imdb_width:
                        w2v_custom = np.array(np.column_stack([self.normalized_w2v(IMDB_dataset.custom_model[w]) for w in text if w in IMDB_dataset.custom_model.wv]))   # normalization, as our custom model has 1.e-3 order values            
                        w2v = np.concatenate((w2v, w2v_custom), axis=1)
            else:
                w2v = w2v_google
            
        w_offset = img.shape[1]-w2v.shape[1]         
        
        if w_offset < 3: # the w2v image is larger than the intended one, truncate it            
            img = w2v[:, 0:img.shape[0]]                  
        else: # else, if it is smaller than offset, we will bring it to the center
            w_offset = int(w_offset/2)-1            
            img[:, w_offset:w_offset+ w2v.shape[1]] = w2v               
            
        img = img.reshape(img.shape[0], img.shape[1], 1)                 
        return img   
    
    
    def get_sample(self, index):                
        sentiment = self.data[index]['sentiment'] 
        sentiment = 'negative' if  sentiment=='neg' else  'posative'  # the hashing number is added to be identified from other keywords in other datasets                     
        img = self.text_to_image_2(self.data[index]['text'])
                  
        return img, sentiment    
    
    
    def __getitem__(self, index):
        index = int(index)       
        img, word_str = self.get_sample(index)        
        ToPIL = transforms.ToPILImage()   
        img = ToPIL(img)
        
        if not(self.cf.H_imdb_scale ==0): # resizing just the height      
            new_w = int(img.size[0]*self.cf.H_imdb_scale/img.size[1])
            if new_w>self.cf.MAX_IMAGE_WIDTH: 
                new_w = self.cf.MAX_IMAGE_WIDTH
            img = img.resize( (new_w, self.cf.H_imdb_scale), Image.ANTIALIAS)                
                
        if self.transform:
            img = self.transform(img)    
        
        target = torch.from_numpy( self.cf.PHOC(word_str, cf = self.cf, mode = 'teximdb')   )
        
        return img, target, word_str, 0

    def __len__(self):
        return len(self.data)
     
    def num_classes(self):        
        return len(self.cf.PHOC('dump', self.cf)) # pasing 'dump' word to get the length


    
    
## Depriciated    
#def text_to_image(self, text):
#        no_of_words=0  
#        img  = np.zeros([len(self.gensim_model['universe']), 
#                         self.cf.W_imdb_width], dtype='float32')       
#        sentences = split_sentences(text) # returns a list of sentences
#        for sent_x in sentences:              
#            sent_x = word_tokenize(sent_x)    
#            ''' sentence_struct = pos_tag(sentence_) # breaking it down to its structure '''
#            
#            for word in sent_x:                  
#                word_vectors = IMDB_dataset.gensim_model.wv    # '''we should take this outside the loop, maybe in self'''    
#                if word in word_vectors.vocab: # and word not in self.stop_words:   
#                    wrd2_vector = IMDB_dataset.gensim_model[word]
#                    img[0:len(wrd2_vector), no_of_words] = wrd2_vector                    
#                    no_of_words += 1
#                if no_of_words > 300: # (self.cf.W_imdb_width-1): 
#                    break
#            else: # else belongs to the for-loop
#                continue # only executed if the inner loop did NOT break
#            break 
#        img = img.reshape(img.shape[0], img.shape[1], 1)                 
#        return img
#
#
