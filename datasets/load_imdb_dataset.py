#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:45:02 2018

@author: malrawi
"""

'''
Sentence cleaning and preprocessing
https://hackernoon.com/word2vec-part-1-fe2ec6514d70

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
from torchnlp.datasets import imdb_dataset
import gensim 
from nltk.corpus import brown
from PIL import Image
from nltk import word_tokenize, pos_tag
from torch.utils.data import Dataset
import numpy as np
from gensim.summarization.textcleaner import split_sentences
import re

#text = word_tokenize("They refuse to permit us to obtain the refuse permit")
#xx = pos_tag(text)
#[('They', 'PRP'), ('refuse', 'VBP'), ('to', 'TO'), ('permit', 'VB'), ('us', 'PRP'),
#('to', 'TO'), ('obtain', 'VB'), ('the', 'DT'), ('refuse', 'NN'), ('permit', 'NN')]


''' Some data clearning '''
def clean_the_text(text):     
    
    text = re.sub("\>|\<|br|/|\\'s|\:|\x85", '', text) # will also replave it's by it
    text = re.sub(r'http\S+', '', text)
    text.replace('\?', ' questionmark')
    text.replace('\?', ' exclamation_mark')            
    text.replace("\\", '')
    return text


class IMDB_dataset(Dataset):
    
    gensim_model = None
        
    def __init__(self, cf, mode='train', transform = None):
        # mode: 'train' or 'test'        
        if mode =='train': # this way, gensim model will be created only once, but shared with all other instances
            if  cf.word_corpus_4_text_understanding == 'Google_news' :
                print('--- Load Google\'s pre-trained Word2Vec model')
                IMDB_dataset.gensim_model = gensim.models.KeyedVectors.load_word2vec_format(cf.folder_of_data + 
                                                                            '/models/GoogleNews-vectors-negative300.bin', binary=True)      
            else:
                print('--- Train Word2Vec model using Brown corpus')
                IMDB_dataset.gensim_model = gensim.models.Word2Vec(brown.sents()) # using Brown corpus
            
        self.cf = cf
        self.mode = mode        
        self.transform = transform    
        if mode=='train': # For some reason, this has to be done this way..train = True, then, test=True!
            self.data = imdb_dataset(directory='/home/malrawi/Desktop/My Programs/all_data/imdb/', train = True)
        else:
            self.data = imdb_dataset(directory='/home/malrawi/Desktop/My Programs/all_data/imdb/', test=True)
        

        
    def get_sample(self, index):
        
        
        img  = np.zeros([len(self.gensim_model['universe']), 
                         self.cf.W_imdb_width], dtype='float32')       
        sentiment = self.data[index]['sentiment'] 
        sentiment = 'negative.135' if  sentiment=='neg' else  'posative.792'  # the hashing number is added to be identified from other keywords in other datasets             
        text  = clean_the_text(self.data[index]['text'] )
        sentences = split_sentences(text) # returns a list of sentences
                          
            
        no_of_words=0  
        for sent_x in sentences:              
            sent_x = word_tokenize(sent_x)    
            ''' sentence_struct = pos_tag(sentence_) # breaking it down to its structure '''
            for word in sent_x:                  
                word_vectors = IMDB_dataset.gensim_model.wv        
                if word in word_vectors.vocab:   
                    wrd2_vector = IMDB_dataset.gensim_model[word]
                    img[0:len(wrd2_vector), no_of_words] = wrd2_vector
                    # img[:, no_of_words] = wrd2_vector
                    no_of_words += 1
                if no_of_words > (self.cf.W_imdb_width-1): 
                    break
            else: # else belongs to the for-loop
                continue # only executed if the inner loop did NOT break
            break 
            
        img = img.reshape(img.shape[0], img.shape[1], 1)                 
        return img, sentiment
        
        
    def __getitem__(self, index):
               
        img, word_str = self.get_sample(index)
        target = self.cf.PHOC(word_str, cf = self.cf)   
        ToPIL = transforms.ToPILImage()   
        img = ToPIL(img)
        
        if not(self.cf.H_imdb_scale ==0): # resizing just the height      
            new_w = int(img.size[0]*self.cf.H_imdb_scale/img.size[1])
            if new_w>self.cf.MAX_IMAGE_WIDTH: 
                new_w = self.cf.MAX_IMAGE_WIDTH
            img = img.resize( (new_w, self.cf.H_imdb_scale), Image.ANTIALIAS)                
                
        if self.transform:
            img = self.transform(img)    
        
        return img, target, word_str, 0

    def __len__(self):
        return len(self.data)
     
    def num_classes(self):        
        return len(self.cf.PHOC('dump', self.cf)) # pasing 'dump' word to get the length


    
    
    



