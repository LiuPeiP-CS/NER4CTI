#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/2 上午11:23
# @Author  : PeiP Liu
# @FileName: BertModel.py
# @Software: PyCharm
import torch
from keras_preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import data_utils

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)
sent_maxlen = 512
label2index = {'B-PER':1}
batch_size = 64


def tokenize_sentence(input_sentence, input_sent_labels): # the sentence is a list of words, ['this','is','a','test']
    tokenized_sentence = []
    labels = []

    for word, word_label in zip(input_sentence, input_sent_labels):
        tokenized_word = tokenizer.tokenize(word)
        num_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend(num_subwords*[word_label])

    return tokenized_sentence, labels # ['all@y'], ['I']——> ['all', '@', 'y'], ['I','I','I']


def tokenized_dataset(dataset_sentences, dataset_labels): # dataset_sentences is a list of upper input_sentence
    processed_token_label = [tokenize_sentence(each_sent, corr_labels) for each_sent, corr_labels in zip(dataset_sentences, dataset_labels)]
    tokenized_text = [pair[0] for pair in processed_token_label] # [['all', '@', 'y'],...]
    tokenized_label = [pair[1] for pair in processed_token_label] # [['I', 'I', 'I'],...]

    return tokenized_text, tokenized_label


def padding_sequence(text, label): # the input is output from tokenized_dataset()
    # tokenizer.tokenize and tokenizer.convert_tokens_to_ids can be integrated to tokenizer.encode_pus
    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in text] # [[231, 4214, ...],...], num from bert vocab
    # truncating='post' will truncate at the end of the sequence, the same means as padding='post'
    padded_input_ids = pad_sequences(input_ids, maxlen= sent_maxlen, dtype='long', value = 0.0,
                                     truncating='post', padding='post')
    padded_labels = pad_sequences([[label2index[token_label] for token_label in sent_labels] for sent_labels in label],
                                  maxlen = sent_maxlen, value = label2index['PAD'], padding='post', dtype = 'long',
                                  truncating='post')
    attention_mask = [[float(word_ids != 0.0)for word_ids in sent_ids]for sent_ids in padded_input_ids]

    return padded_input_ids, padded_labels, attention_mask


def split_dataset(padded_input_ids, padded_labels, attention_mask): # the input is output from padding_sequence()
    train_ids, train_labels, train_att_mask, val_ids, val_labels, val_att_mask = \
        train_test_split(padded_input_ids, padded_labels, attention_mask, random_state=1024, test_size=0.1)
    # convert the
    train_ids = torch.tensor(train_ids)
    train_labels = torch.tensor(train_labels)
    train_att_mask = torch.tensor(train_att_mask)
    val_ids = torch.tensor(val_ids)
    val_labels = torch.tensor(val_labels)
    val_att_mask = torch.tensor(val_att_mask)

    train_data = TensorDataset(train_ids, train_labels, train_att_mask) # there can be several params
    # On the basis of map style (i.e. you can get the value by index),
    # we can make the "shuffle=True" in DataLoader to replace RandomSampler, while default shuffle is False.
    # train_dataloader = DataLoader(train_data, shuffle = True, batch_size = batch_size)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_ids, val_labels, val_att_mask)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader