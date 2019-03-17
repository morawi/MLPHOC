#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sat Jan 26 15:01:58 2019

@author: malrawi


"""

from string import punctuation
from collections import Counter
from nltk.corpus import stopwords

from gensim.models import Word2Vec

 
 
# turn a doc into clean tokens
def clean_text_3(text):
	# split into tokens by white space
    text = text.lower()
    tokens = text.split()
	# remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def build_w2v_model(reviews):
    min_occurane = 5
    # define vocab
    vocab = Counter()
    for review in reviews:
        tokens = clean_text_3(review)        
        vocab.update(tokens)     
    
    # print(vocab.most_common(50)) # print the top words in the vocab        
    tokens = [k for k, c in vocab.items() if c >= min_occurane] # keep tokens with > 5 occurrence    
    model = Word2Vec([tokens], min_count=1, size = 300)    
    return model


all_reviews = ['This has always been one of my dish; This has always been one of my dish',
               'Over the last few years I have become; Over the last few years I have become; This has always been one of my dish ']
model = build_w2v_model(all_reviews)

# model = Word2Vec([tokens], min_count=1, size = 300)
# summarize the loaded model
# print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['always'])
# save model
# model.save('model.bin')
# load model
# new_model = Word2Vec.load('model.bin')


