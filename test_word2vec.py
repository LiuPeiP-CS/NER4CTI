#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 下午7:52
# @Author  : PeiP Liu
# @FileName: test_word2vec.py
# @Software: PyCharm

import gensim
import sys
import pickle

data = [['the', 'full', 'model', 'can', 'be', ' stored', 'loaded', 'via', 'its', 'save', 'and', 'load', 'methods'
        , 'the', 'trained', 'word', 'vectors', 'can', 'also', 'be', 'stored', 'loaded', 'from', 'a',
        'format', 'compatible', 'with', 'the', 'original', 'word2vec', 'implementation', 'via', 'and']]
# min_count under the args.index2word, and the hid_dim in lstm
model = gensim.models.Word2Vec(data, min_count=1, size=5, window=4)  # CBOW

model.save('/home/liupei/test_word2vec.bin')

test_dict = dict()
# word_list = list(model.wv.vocab)
# print(word_list)
# for word in word_list:
#     test_dict.update({word: model.wv.vectors[model.wv.vocab[word].index]})
print(len(model.wv.index2word))
print(len(model.wv.vocab))
print(model.wv.vectors)
for word, embedding in zip(model.wv.index2word, model.wv.vectors):  # test_dict
    print(word, embedding)
    test_dict.update({word: embedding})
with open('/home/liupei/test_dict.pickle', 'wb') as file:
    pickle.dump(test_dict, file)


model = gensim.models.Word2Vec.load('/home/liupei/test_word2vec.bin')
sim_words = model.most_similar('full', topn=3)
for sim_word in sim_words:
    print(model.wv.vectors[model.wv.vocab[sim_word[0]].index])
    print(model.wv.vocab[sim_word[0]].index)
