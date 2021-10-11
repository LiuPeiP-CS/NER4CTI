#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/16 下午3:43
# @Author  : PeiP Liu
# @FileName: arguments.py
# @Software: PyCharm
import os
import torch
import pickle
import numpy as np

# import sys
# sys.path.append(".")

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
ab_path = '/bigdata/liupeipei/Multi_features_based_semantic_augmentation_networks_for_NER_in_TI'


def read_pickle(file_add):
    with open(file_add, 'rb') as file:
        data = pickle.load(file)
    return data


def read_numpy(file_addr):
    np_data = np.load(file_addr)
    return np_data


class BasicArgs:
    batch_size = 64
    max_seq_len = 256
    learning_rate = 5e-3
    # choose the device, if GPU is available, we can use it. otherwise, the cpu is a replacement
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    total_train_epoch = 400

    orig_dataset = read_pickle(ab_path + '/Result/Data/MalwareDB/orig_dict.pickle')
    num_train = orig_dataset['num_train']
    num_valid = orig_dataset['num_valid']
    num_test = orig_dataset['num_test']

    dict_dataset = read_pickle(ab_path + '/Result/Data/MalwareDB/index_dict.pickle')
    label2idx = dict_dataset['label2index']  # for BERT loss
    idx2label = dict_dataset['index2label']  # for prediction
    word2idx = dict_dataset['word2index']
    idx2word = dict_dataset['index2word']
    pos2idx = dict_dataset['pos2index']  # for pos_emb training
    idx2pos = dict_dataset['index2pos']
    case2idx = dict_dataset['case2idx']

    num_labels = len(label2idx)


class BertArgs(BasicArgs):
    output_dir = ab_path + '/Result/BERT_model/MalwareDB/'

    train_seq_list = BasicArgs.orig_dataset['train_sentences']  # for BERT train, list type
    train_seq_label_list = BasicArgs.orig_dataset['train_labels']  # for BERT train, list type
    valid_seq_list = BasicArgs.orig_dataset['valid_sentences']  # for BERT valid, list type
    valid_seq_label_list = BasicArgs.orig_dataset['valid_labels']  # for BERT valid, list type
    test_seq_list = BasicArgs.orig_dataset['test_sentences']  # for BERT test, list type
    test_seq_label_list = BasicArgs.orig_dataset['test_labels']  # for BERT test, list type

    model_list = ['bert-base-uncased', 'bert-large-uncased', 'bert-base-cased']
    model_id = 2

    # if you retrain the model, please make load_checkpoint = True
    load_checkpoint = False
    weight_decay_finetune = 1e-5
    lr_crf_fc = 1e-5
    weight_decay_crf_fc = 1e-5
    warmup_proportion = 0.002

    # the larger batch_size is, the effect will be better. However, some labs's GPU is not available enough.
    # So we can accumulate X*gradient (gradient in each batch) to achieve the same effect as batch_size*X.
    # that is, we don't empty the gradient each batch until X batches
    # rf https://blog.csdn.net/Princeicon/article/details/108058822
    gradient_accumulation_steps = 4


class BilstmCnnArgs(BasicArgs):
    output_dir = ab_path + '/Result/LSTM_model/MalwareDB/'

    # all_sentences = BasicArgs.orig_dataset['all_sentences']  # for word2vec training

    real_sent_maxlen = BasicArgs.orig_dataset['sent_maxlen']  # list type
    real_word_maxlen = BasicArgs.orig_dataset['word_maxlen']  # list type
    # real_sent_maxlen = 128
    # real_word_maxlen = 100

    train_id_pad = read_pickle(ab_path + '/Result/Data/MalwareDB/train_id_pad_dict.pickle')  # used for bilstm train
    train_wordids_pad = train_id_pad['train_wordids_pad']
    train_charids_pad = train_id_pad['train_charids_pad']
    train_labelids_pad = train_id_pad['train_labelids_pad']
    train_posids_pad = train_id_pad['train_posids_pad']
    train_caseids_pad = train_id_pad['train_caseids_pad']

    valid_id_pad = read_pickle(ab_path + '/Result/Data/MalwareDB/valid_id_pad_dict.pickle')  # used for bilstm validation
    valid_wordids_pad = valid_id_pad['valid_wordids_pad']
    valid_charids_pad = valid_id_pad['valid_charids_pad']
    valid_labelids_pad = valid_id_pad['valid_labelids_pad']
    valid_posids_pad = valid_id_pad['valid_posids_pad']
    valid_caseids_pad = valid_id_pad['valid_caseids_pad']

    test_id_pad = read_pickle(ab_path + '/Result/Data/MalwareDB/test_id_pad_dict.pickle')  # used for bilstm test
    test_wordids_pad = test_id_pad['test_wordids_pad']
    test_charids_pad = test_id_pad['test_charids_pad']
    test_labelids_pad = test_id_pad['test_labelids_pad']
    test_posids_pad = test_id_pad['test_posids_pad']
    test_caseids_pad = test_id_pad['test_caseids_pad']

    # all_posids_pad = train_posids_pad + valid_posids_pad + test_posids_pad  # used for pos_emb training

    word_emb_table = read_numpy(ab_path + '/Result/Embedding/MalwareDB/word_embedding.npy')  # numpy.array
    char_emb_table = read_numpy(ab_path + '/Result/Embedding/MalwareDB/char_embedding.npy')
    pos_emb_table = read_numpy(ab_path + '/Result/Embedding/MalwareDB/pos_embedding.npy')
    case_emb_table = read_numpy(ab_path + '/Result/Embedding/MalwareDB/case_embedding.npy')

    char_embed_dim = 30
    word_embed_dim = 50
    pos_embed_dim = 10
    case_embed_dim = 8
    input_dim = 98  # char_emb_dim + word_emb_dim + pos_emb_dim + case_emb_dim
    hid_dim = 256
    model_dim = 128

    transformer_num_blocks = 4
    transformer_num_heads = 8

    index2word = read_pickle(ab_path + '/Result/Data/MalwareDB/index_dict.pickle')['index2word']
    word2vec = read_numpy(ab_path + '/Result/Embedding/MalwareDB/word2vec_embedding.npy')
    sim_num = 4

    bilstm_layers = 4

    dropout_rate = 0.5
    attention_type = 'general'

    min_delta = 0.0001
    patience = 12

    lr = 5e-4
    weight_decay = 0.001
    min_lr = 5e-5
    lr_decay_factor = 0.5

    word_pad_indx = 0
    label_bos_indx = 0
    label_eos_indx = 1
    label_pad_indx = 2
