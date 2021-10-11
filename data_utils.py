#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 下午5:21
# @Author  : PeiP Liu
# @FileName: data_utils.py
# @Software: PyCharm

import os
from collections import Counter
import numpy as np
from numpy import random
import stanza
# import nltk
# rf https://stanfordnlp.github.io/stanza/pos.html


def processing_orgdata(mode):
    """
    read the word from file, and build sentence. every line contains a word and it's tag.
    every sentence is splitted by an empty line.
    :param mode: the data will be used for which mode
    :return:
    """
    sentences = []
    sentences_labels = []
    sentences_pos = []
    str_sentences = []

    sent_maxlen = 0
    word_maxlen = 0

    sentence = []
    sentence_label = []
    str_sent = ''

    data_dir = 'MalwareDB/MalwareDB/'
    file_list = ['train.txt', 'valid.txt', 'test.txt']
    if mode == 'train':
        file_read = open(os.path.join(data_dir, file_list[0]), 'r')
    elif mode == 'valid':
        file_read = open(os.path.join(data_dir, file_list[1]), 'r')
    else:
        file_read = open(os.path.join(data_dir, file_list[2]), 'r')

    # this is used for the pos_tag
    nlp = stanza.Pipeline('en', processors='tokenize,pos')

    for line in file_read:
        line = line.strip()
        if line == '':
            if not sentence:
                continue
            # parser pos_tag for every word
            sentence_pos = []
            doc = nlp(str_sent)
            for sent in doc.sentences:
                for word in sent.words:
                    sentence_pos.append(word.pos)
            # following is the case of nltk, its return is like [('This', 'DT'), ('is', 'VBZ'), ('my', 'PRP$'),...]
            # token_sent = nltk.word_tokenize(str_sent)
            # sentence_pos = nltk.pos_tag(token_sent)
            sentences_pos.append(sentence_pos)

            sent_maxlen = max(len(sentence), sent_maxlen)

            assert len(sentence_label) == len(sentence)
            sentences.append(sentence)
            sentences_labels.append(sentence_label)
            str_sentences.append(str_sent)

            sentence = []
            sentence_label = []
            str_sent = ''
        else:
            word_label = line.split()
            if len(word_label) != 2:
                continue
            sentence.append(word_label[0])  # a list which is a orig sentence
            sentence_label.append(word_label[1])  # the label list corresponding to orig sentence
            word_maxlen = max(word_maxlen, len(str(word_label[0])))
            str_sent = str_sent + ' ' + str(word_label[0])

    num_sentence = len(sentences)
    print("The mode is : {}".format(mode))
    print("sent max length is {}".format(sent_maxlen))
    print("word max length is %d" % word_maxlen)
    print('num of sentences is ', num_sentence)
    return sentences, sentences_labels, sentences_pos, str_sentences, sent_maxlen, word_maxlen, num_sentence


def build_vocab(sentences, sentences_labels, sentences_pos): # 这里的输入信息都是train+valid+test

    word_list = []
    char_list = []
    label_list = []
    pos_list = []
    for i_sent in range(len(sentences)):
        for j_word in range(len(sentences[i_sent])):
            word_list.append(sentences[i_sent][j_word].strip())  # all the words in dict can also be lower
            label_list.append(sentences_labels[i_sent][j_word])
            pos_list.append(sentences_pos[i_sent][j_word])
            for char in sentences[i_sent][j_word]:
                char_list.append(char)

    # word_set = list(set(word_list))
    word_counter = Counter(word_list)
    word_set = [word[0] for word in word_counter.most_common() if word[1] >= 2]  # make sure the number of word>=2
    print("The word set is ", word_set)
    word2index = {each_word: word_index+2 for word_index, each_word in enumerate(word_set)}
    word2index['[PAD]'] = 0; word2index['[UNK]'] = 1
    index2word = {word_index: each_word for each_word, word_index in word2index.items()}

    char_counter = Counter(char_list)
    char_set = [char[0] for char in char_counter.most_common() if char[1] >= 2]  # make sure the number of char>=2
    print("The char set is ", char_set)
    char2index = {each_char: char_index+2 for char_index, each_char in enumerate(char_set)}
    char2index['[CPAD]'] = 0; char2index['[CUNK]'] = 1
    index2char = {char_index: each_char for each_char, char_index in char2index.items()}

    label_counter = Counter(label_list)
    label_set = [label_item[0] for label_item in label_counter.most_common()]
    print("The label set is ",label_set)
    label2index = {each_label: label_index + 3 for label_index, each_label in enumerate(label_set)}
    label2index['[BOS]'] = 0;  label2index['[EOS]'] = 1; label2index['[X]'] = 2
    index2label = {label_index: each_label for each_label, label_index in label2index.items()}

    pos_set = list(set(pos_list))
    print('The pos set is ', pos_set)
    pos2index = {pos : id+1 for id, pos in enumerate(pos_set)}
    pos2index['[PPAD]'] = 0
    index2pos = {id:pos for pos, id in pos2index.items()}
    return word2index, index2word, char2index, index2char, label2index, index2label, pos2index, index2pos


def case_feature():
    case2idx = {'allNum':0, 'allLower':1, 'allUpper':2, "upperInit":3, 'other':4, 'main_num':5, 'contain_num':6, '[PAD]':7}
    case_emb = np.identity(len(case2idx), dtype='float32')
    return case2idx, case_emb


def get_token_case(token, case2idx): # 组成token的字符形态学特征
    num_digits = 0
    for char in token.strip(): # 加上前后处理，防止之前的处理不完全
        if char.isdigit():
            num_digits = num_digits + 1
    digit_prop = num_digits / float(len(token))

    casing = 'other'
    if token.isdigit():
        casing = 'allNum'
    elif digit_prop > 0.5:
        casing = 'main_num'
    elif token.islower():
        casing = 'allLower'
    elif token.isupper():
        casing = 'allUpper'
    elif token.istitle():
        casing = 'upperInit'
    elif num_digits > 0:
        casing = 'contain_num'

    return case2idx[casing]


def text2ids(sentences, sentences_labels, sentences_pos, word2index, char2index, label2index, pos2index, case2idx):# 这里的输入信息分别是train、valid、test等
    sents_wordids = []
    sents_charids = []
    sents_labels_ids = []
    sents_posids = []
    sents_caseids = []

    for sent_iter, sent in enumerate(sentences):
        word_ids = []  # convert the sentence to token_id
        char_ids = []  # convert the sentence to char_id
        label_ids = []  # convert the sentence_label to label_id
        pos_ids = []  # convert the sentence_pos to pos_id
        case_ids = []  # get the morphology_id
        for word_iter, word in enumerate(sent):
            if word in word2index:
                wordid = word2index[word]
            elif word.lower() in word2index:
                wordid = word2index[word.lower()]  # use the lower of the word
            else:
                wordid = word2index['[UNK]']  # the low frequency words and OOV

            charid = []  # the chars of a word
            for char in word:
                if char not in char2index:
                    charid.append(char2index['[CUNK]'])  # the low frequency chars and OOV
                else:
                    charid.append(char2index[char])

            word_ids.append(wordid)
            char_ids.append(charid)
            label_ids.append(label2index[sentences_labels[sent_iter][word_iter]])
            pos_ids.append(pos2index[sentences_pos[sent_iter][word_iter]])
            case_ids.append(get_token_case(word, case2idx))

        sents_wordids.append(word_ids)
        sents_charids.append(char_ids)
        sents_labels_ids.append(label_ids)
        sents_posids.append(pos_ids)
        sents_caseids.append(case_ids)

    return sents_wordids, sents_charids, sents_labels_ids, sents_posids, sents_caseids


def sentence_padding(sentences, sent_maxlen, padding_value): # 这里，每个句子中的词已经转换成了词索引
    padded = []
    actual_len = []
    for sent in sentences:
        if len(sent) < sent_maxlen:
            padded.append(sent + [padding_value] * (sent_maxlen-len(sent)))
            # np.pad(sent,pad_width=(0, sent_maxlen-len(x)),mode='constant',constant_values=padding_value)
            actual_len.append(len(sent))
        else:
            padded.append(sent[:sent_maxlen])
            actual_len.append(sent_maxlen)
    return padded
    # return np.array(padded), actual_len


def char_sentences_padding(sents_charids, sent_maxlen, word_maxlen): # padding_value是char2index['[CPAD]'] = 0
    pad_char_sentences = []
    for sent in sents_charids:
        sent_char_pad = np.zeros([sent_maxlen, word_maxlen], dtype = np.int32) # 表示一个句子
        sc_pad = []  # one sequence
        for word in sent:  # a sequence of char_id from char2indx
            char_pad = np.zeros([word_maxlen], dtype=np.int32)  # on word
            if len(word) <= word_maxlen:
                char_pad[:len(word)] = word
            else:
                char_pad = word[:word_maxlen]
            # char_pad = word[:word_maxlen] + [padding_value] * max(word_maxlen - len(word), 0)

            sc_pad.append(char_pad)  # a list of char_id for a sentence

        for i in range(len(sc_pad)):
            sent_char_pad[i, :len(sc_pad[i])] = sc_pad[i]  # post padding
            # sent_char_pad[sent_maxlen-len(sc_pad)+i, :len(sc_pad[i])] = sc_pad[i] # trunte padding

        pad_char_sentences.append(sent_char_pad)  # the list of padded sentences

    return pad_char_sentences
    # return np.array(pad_char_sentences)  # numpy array


def build_word_emb_table(index2word, glove_embed_dict, word_embed_dim):
    scale = np.sqrt(3.0 / word_embed_dim)
    word_emb_table = np.empty([len(index2word), word_embed_dim], dtype=np.float32)
    word_emb_table[:2, :] = np.random.uniform(-scale, scale, [2, word_embed_dim])  # UNK and PAD
    for index, word in index2word.items():
        if word in glove_embed_dict:
            word_emb = glove_embed_dict[word]
        elif word.lower() in glove_embed_dict:
            word_emb = glove_embed_dict[word.lower()]
        else:
            word_emb = np.random.uniform(-scale, scale, [1, word_embed_dim])
        word_emb_table[index, :] = word_emb
    return word_emb_table


def build_char_emb_table(index2char, char_embed_dim=30):
    scale = np.sqrt(3.0/char_embed_dim)
    char_emb_table = np.random.uniform(-scale, scale, [len(index2char), char_embed_dim]).astype(np.float32)
    return char_emb_table


def build_pos_emb_table():
    pos_emb_table = np.load('Result/PosEmbedding/MalwareDB/pos_embedding.npy')
    return pos_emb_table

'''
def split_wordlabelpos(input_sentences):# [[['This', 'B-PER', 'NN'], ...],...]
    sent_word = [[word[0] for word in sent] for sent in input_sentences] # [['this','is','a','test'],...]
    sent_label = [[word[1] for word in sent] for sent in input_sentences] # it is like the former, but only by 'B-PER'
    sent_pos = [[word[2] for word in sent] for sent in input_sentences]
    return [sent_word, sent_label, sent_pos]

def word2charlist(char2index, index2word): #＃＃＃＃＃＃＃ 后面要注意填充词和未登录词　＃＃＃＃＃＃＃＃＃＃
    word2charids = {}
    for ind, word in index2word.items():
        char_ids = []
        for char in word:
            if char in char2index.keys():
                char_ids = char_ids + [char2index[char]]
            else:
                char_ids = char_ids + [char2index['[CUNK]']]
        word2charids[ind] = char_ids
    return word2charids # 返回值是｛词索引：字符索引列表｝的形式

def word_padding(word2charids, word_maxlen, padding_value): # 输入形式是｛词索引：字符索引列表｝
    word_padded = dict()
    word_actuallen = dict()
    for word_idx, charids_list in word2charids.items():
        charids_list_ = charids_list[:word_maxlen] + [padding_value] * max(word_maxlen-len(charids_list), 0)
        word_padded[word_idx] = charids_list_
        word_actuallen[word_idx] = min(len(charids_list), word_maxlen)
    return word_padded, word_actuallen # 输出是｛词索引：扩充后的字符索引列表｝，｛词索引：实际包含的字符数量｝
    
def get_batch(dataset, batch_size, shuffle=False):
    data_size = len(dataset)
    num_batch = int((data_size-1) / batch_size) + 1
    if shuffle:
        indices = np.random.permutation(np.arange(data_size))
        data_shuffle = np.array(dataset)[indices]
    else:
        data_shuffle = np.array(dataset)

    for i_batch in num_batch:
        start_id = i_batch * batch_size
        end_id = min((i_batch + 1) * batch_size, data_size)
        yield data_shuffle[start_id:end_id]
'''


def gen_batch_data(sents, labels, bert_sents, bert_labels, num_sentence, batch_size):
    word_sentences = np.array(sents[0])  # wordids_pad
    char_sentences = np.array(sents[1])  # charids_pad
    pos_sentences = np.array(sents[2])
    case_sentences = np.array(sents[3])
    labels_sentences = np.array(labels)

    bert_sents = np.array(bert_sents)
    bert_labels = np.array(bert_labels)

    data_idx = np.arange(num_sentence)
    random.shuffle(data_idx)

    i = 0
    while True:
        if i + batch_size >= num_sentence:
            batch_inx = data_idx[i:]
            batch_word_sentences = word_sentences[batch_inx]
            batch_char_sentences = char_sentences[batch_inx]
            batch_pos_sentencens = pos_sentences[batch_inx]
            batch_case_sentences = case_sentences[batch_inx]
            batch_labels_sentences = labels_sentences[batch_inx]

            batch_bert_sents = bert_sents[batch_inx]
            batch_bert_labels = bert_labels[batch_inx]

            yield (batch_word_sentences, batch_char_sentences, batch_pos_sentencens, batch_case_sentences), \
                  batch_labels_sentences, batch_bert_sents, batch_bert_labels
            break
        else:
            batch_inx = data_idx[i: i+batch_size]
            batch_word_sentences = word_sentences[batch_inx]
            batch_char_sentences = char_sentences[batch_inx]
            batch_pos_sentencens = pos_sentences[batch_inx]
            batch_case_sentences = case_sentences[batch_inx]
            batch_labels_sentences = labels_sentences[batch_inx]

            batch_bert_sents = bert_sents[batch_inx]
            batch_bert_labels = bert_labels[batch_inx]

            yield (batch_word_sentences, batch_char_sentences, batch_pos_sentencens, batch_case_sentences), \
                  batch_labels_sentences, batch_bert_sents, batch_bert_labels
            i = i + batch_size


class DataLoader:
    def __init__(self, input_data, labels):
        self.input_data = input_data
        self.labels = labels
        self.num_data = len(input_data)
        self.indexes = np.arange(self.num_data)

    def get_batch(self, batch_size, shuffle = True):
        if shuffle:
            np.random.shuffle(self.indexes)

        iter = 0
        while True:
            if iter + batch_size >= self.num_data:
                yield self.input_data[self.indexes[iter:]], self.labels[self.indexes[iter:]]
                break
            else:
                yield self.input_data[self.indexes[iter:iter+batch_size]], self.labels[self.indexes[iter:iter+batch_size]]
                iter = iter + batch_size

    def __len__(self):
        return self.num_data


class GloveFeature:
    def __init__(self, glove_path):
        self.glove_path = glove_path
        self.glove_token2inx, self.glove_dim = self.glove_vocab()

    def glove_vocab(self):
        vocab = set()
        embed_dim = -1
        with open(self.glove_path, 'r') as file_read:
            for line in file_read:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split(' ')
                if embed_dim < 0:
                    embed_dim = len(tokens) - 1
                else:
                    assert (embed_dim + 1 == len(tokens))
                word = tokens[0]
                vocab.add(word)
        print('glove vocab done. {} tokens'.format(len(vocab)))
        glove_token2inx = {token: ind for ind, token in enumerate(vocab)}
        return glove_token2inx, embed_dim

    def load_glove_embedding(self):
        file_read = open(self.glove_path, 'r')
        # glove_embeddings = np.random.random([len(self.glove_token2inx), self.glove_dim])
        embedding_dict = dict()
        for line in file_read:
            line = line.strip()
            if len(line) == 0:
                continue
            line = line.split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            embedding_dict[word] = np.array(embedding)

        return embedding_dict

