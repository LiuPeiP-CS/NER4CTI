#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 下午4:04
# @Author  : PeiP Liu
# @FileName: wod2vec_embedding.py
# @Software: PyCharm

import gensim
# import gensim.models
import numpy as np
import pickle


file_add = '../Result/Data/MalwareDB/orig_dict.pickle'
with open(file_add, 'rb') as file:
	dataset = pickle.load(file)
all_sentences = dataset['all_sentences']

data = []
for sent in all_sentences:
    temp = []
    for word in sent:
        temp.append(word.strip())
    data.append(temp)

# min_count under the args.index2word, and the hid_dim in lstm
model = gensim.models.Word2Vec(data, min_count=2, size=256, window=3)  # CBOW
# model2 = gensim.models.Word2Vec(data, min_count=3, size=512, window=4, sg=1)  # Skip-Gram

model.save('../Result/Embedding/MalwareDB/word2vec_embedding.bin')

# method_1
'''
word2vec_dict = dict()
word_list = list(model.wv.vocab)
for word in word_list:
    word2vec_dict.update({word: model.wv.vectors[model.wv.vocab[word].index]})
'''

# method_2
'''
word2vec_dict = dict()
for word, embedding in zip(model.wv.index2word, model.wv.vectors):
    word2vec_dict.update({word: embedding})
'''
# # one-to-one correspondence between model.wv.index2word and model.wv.vectors
np.save('../Result/Embedding/MalwareDB/word2vec_embedding.npy', model.wv.vectors)

