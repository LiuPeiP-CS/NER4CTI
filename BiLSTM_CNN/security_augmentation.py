#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 上午11:01
# @Author  : PeiP Liu
# @FileName: security_augmentation.py
# @Software: PyCharm
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
# from sklearn.utils.extmath import softmax
import gensim
import random


class BaseInternalAugmentation:

    def __init__(self, args):
        self.args = args
        self.word2vec = args.word2vec
        # self.all_word_embs = np.array(list(self.word2vec.values()))
        self.wv_model = gensim.models.Word2Vec.load('Result/Embedding/MalwareDB/word2vec_embedding.bin')

    def consine_sim(self, word):
        sim_scores = []
        sim_words = self.wv_model.wv.most_similar(word, topn=self.args.sim_num)
        for sim_word in sim_words:
            sim_scores.append(sim_word[1])
        return sim_scores

        # sub_emb = np.array(self.word2vec[word])
        # similarity_scores = sub_emb.dot(self.all_word_embs) / (np.linalg.norm(sub_emb, axis=1)
        #                                                        * np.linalg.norm(self.all_word_embs))  # consine
        # return similarity_scores

    def most_k_index(self, word):
        words_index = []
        sim_words = self.wv_model.wv.most_similar(word, topn=self.args.sim_num)
        for sim_word in sim_words:
            words_index.append(self.wv_model.wv.vocab[sim_word[0]].index)
        return words_index

        # simscore_list = np.array(self.consine_sim(word))  # the similarities between word and all other words
        # the index of most similar K words
        # new_indx = np.argpartition(simscore_list, -self.args.sim_num)[-self.args.sim_num:]
        # return new_indx

    def compute_atten_embedding(self, word):
        sim_scores = np.array(self.consine_sim(word), dtype=np.float32)
        e_x = np.exp(sim_scores - np.max(sim_scores))
        f_x = e_x / e_x.sum()  # the softmax weight

        similar_words_embedding = self.word2vec[self.most_k_index(word)] # the index of most similar K words and vectors
        hard_atten_embedding = np.sum(f_x[:, None] * similar_words_embedding, axis=0)
        return hard_atten_embedding

        # sim_scores = self.consine_sim(word)
        # inds = self.most_k_index(word)
        # most_k_similar = sim_scores[inds]  # 最相似的k个数的相似程度
        # y = np.exp(most_k_similar - np.max(most_k_similar))
        # f_x = y / np.sum(np.exp(most_k_similar))  # 根据相似程度求解softmax
        # similar_embeddings = self.all_word_embs[inds]  # 最相似的k个元素的词向量
        # hard_atten_embedding = np.sum(f_x * similar_embeddings, axis=-1)
        # return hard_atten_embedding


class HardInternalAugmentation(BaseInternalAugmentation):
    def __init__(self, args):
        super(HardInternalAugmentation, self).__init__(args)

    def hard_augmentation_embedding(self, lstm_dim):
        attention_embedding = np.zeros([len(self.args.idx2word), lstm_dim], dtype=np.float32)
        for ind, word in self.args.idx2word.items():
            if word in self.wv_model.wv.index2word:
                attention_embedding[ind, :] = self.compute_atten_embedding(word)
            else:
                # get the random augmentation embedding
                scale = np.sqrt(3.0 / lstm_dim)
                attention_embedding[ind, :] = np.random.uniform(-scale, scale, [1, lstm_dim])

        hard_attention_embedding = torch.tensor(attention_embedding, dtype=torch.float32).to(self.args.device)
        return hard_attention_embedding


class SoftInternalAugmentation(BaseInternalAugmentation):
    def __init__(self, args):
        super(SoftInternalAugmentation, self).__init__(args)

    def soft_augmentation_words(self, batch_word_data):  # get the index of most similar K words
        sims = np.zeros([len(self.args.index2word), self.args.sim_num], dtype=np.int32)
        for ind, word in self.args.index2word.items():
            if word in self.wv_model.wv.index2word:
                sims[ind, :] = self.most_k_index(word)
            else:
                sims[ind, :] = random.sample(range(0, len(self.wv_model.wv.vocab)), self.args.sim_num)   # 注意此处的处理

        sims = torch.from_numpy(sims).long().to(self.args.device)  # move to device

        similar_words = sims[batch_word_data]
        return similar_words  # (batch_size, sent_maxlen, sim_num)


class SoftAugmentationAttention(nn.Module):
    def __init__(self, word2vec, emb_size, dropout_rate):
        super(SoftAugmentationAttention, self).__init__()
        self.k_augment = nn.Embedding.from_pretrained(torch.tensor(word2vec, dtype=torch.float32), freeze=False)
        assert word2vec.shape[-1] == emb_size
        self.weight = nn.Parameter(torch.empty(emb_size, emb_size), requires_grad=True)
        nn.init.xavier_uniform_(self.weight)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_feature, similar_words_sent):
        k_sims_feature = self.k_augment(similar_words_sent)  # (batch_size, sent_maxlen, sim_num, emb_size)
        batch_size, sent_len, sim_num, emb_size = k_sims_feature.shape
        k_sims_feature = k_sims_feature.view(batch_size*sent_len, sim_num, emb_size)
        q_feature = hidden_feature.reshape(batch_size*sent_len, 1, emb_size)

        attention_matrix = q_feature.matmul(self.weight).bmm(k_sims_feature.transpose(1, 2))

        soft_attention = torch.softmax(attention_matrix, dim=-1)
        soft_attention = self.dropout(soft_attention)

        output = soft_attention.bmm(k_sims_feature).view(batch_size, sent_len, emb_size)
        output = self.dropout(output)
        return output
