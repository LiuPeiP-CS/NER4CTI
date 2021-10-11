#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/1 上午10:52
# @Author  : PeiP Liu
# @FileName: data_processing.py
# @Software: PyCharm

import torch
import torch.nn as nn
import pickle
import numpy as np
from data_utils import *

# get the text data from orig_file
train_dataset = processing_orgdata('train')
valid_dataset = processing_orgdata('valid')
test_dataset = processing_orgdata('test')

# all_str_sentences = train_dataset[3] + valid_dataset[3] + test_dataset[3]
# all_str_sentences = train_dataset[3] + valid_dataset[3] + test_dataset[3]
# with open('Result/Data/sentences_text', 'a+') as file:
#     for sent in all_str_sentences:
#         file.write(sent+'\n')
# file.close()

# get the max length of sentence(word) and word(char)
sent_maxlen = max(train_dataset[4], valid_dataset[4], test_dataset[4]) # 这里的最长后续要根据数据的分布情况调整
word_maxlen = max(train_dataset[5], valid_dataset[5], test_dataset[5]) # 这里的最长后续要根据数据的分布情况调整

# construct the dict of
build_vocab_sentences = train_dataset[0] + valid_dataset[0] + test_dataset[0]
build_vocab_sentences_labels = train_dataset[1] + valid_dataset[1] + test_dataset[1]
build_vocab_sentences_pos = train_dataset[2] + valid_dataset[2] + test_dataset[2]

# create the dict for storing dataset
orig_dict = dict(all_sentences=build_vocab_sentences,
                 all_sentences_labels=build_vocab_sentences_labels,
                 all_sentences_pos=build_vocab_sentences_pos,
                 train_sentences=train_dataset[0],
                 valid_sentences=valid_dataset[0],
                 test_sentences=test_dataset[0],
                 train_labels=train_dataset[1],
                 valid_labels=valid_dataset[1],
                 test_labels=test_dataset[1],
                 train_pos=train_dataset[2],
                 valid_pos=valid_dataset[2],
                 test_pos=test_dataset[2],
                 sent_maxlen=sent_maxlen,
                 word_maxlen=word_maxlen,
                 num_train=train_dataset[-1],
                 num_valid=valid_dataset[-1],
                 num_test=test_dataset[-1]
                 )

with open('Result/Data/MalwareDB/orig_dict.pickle', 'wb') as file:
    pickle.dump(orig_dict, file)  # can be used for bert

# the dict result，后续我们需要增强字符字典的内容
build_vocab_result = build_vocab(build_vocab_sentences, build_vocab_sentences_labels, build_vocab_sentences_pos)
case2idx, case_emb = case_feature()

# create the index dict
index_dict = dict(word2index=build_vocab_result[0],
                  index2word=build_vocab_result[1],
                  char2index=build_vocab_result[2],
                  index2char=build_vocab_result[3],
                  label2index=build_vocab_result[4],
                  index2label=build_vocab_result[5],
                  pos2index=build_vocab_result[6],
                  index2pos=build_vocab_result[7],
                  case2idx=case2idx
                  )

with open('Result/Data/MalwareDB/index_dict.pickle', 'wb') as file:
    pickle.dump(index_dict, file)  # can be used for others

# convert the orig_train_text to id
train_text2ids = text2ids(train_dataset[0], train_dataset[1],
                          train_dataset[2], build_vocab_result[0],build_vocab_result[2],
                          build_vocab_result[4], build_vocab_result[6], case2idx)

train_id_dict = dict(train_sents_wordids=train_text2ids[0],
                     train_sents_charids=train_text2ids[1],
                     train_sents_labelids=train_text2ids[2],
                     train_sents_posids=train_text2ids[3],
                     train_sents_caseids=train_text2ids[4]
                     )

with open('Result/Data/MalwareDB/train_id_dict.pickle', 'wb') as file:
    pickle.dump(train_id_dict, file)  # can be used for others

# convert the orig_valid_text to id
valid_text2ids = text2ids(valid_dataset[0], valid_dataset[1],
                          valid_dataset[2], build_vocab_result[0], build_vocab_result[2],
                          build_vocab_result[4], build_vocab_result[6], case2idx)

valid_id_dict = dict(valid_sents_wordids=valid_text2ids[0],
                     valid_sents_charids=valid_text2ids[1],
                     valid_sents_labelids=valid_text2ids[2],
                     valid_sents_posids=valid_text2ids[3],
                     valid_sents_caseids=valid_text2ids[4]
                     )

with open('Result/Data/MalwareDB/valid_id_dict.pickle', 'wb') as file:
    pickle.dump(valid_id_dict, file)  # can be used for others

# convert the orig_test_text to id
test_text2ids = text2ids(test_dataset[0], test_dataset[1],
                         test_dataset[2], build_vocab_result[0], build_vocab_result[2],
                         build_vocab_result[4], build_vocab_result[6], case2idx)

test_id_dict = dict(valid_sents_wordids=test_text2ids[0],
                    valid_sents_charids=test_text2ids[1],
                    valid_sents_labelids=test_text2ids[2],
                    valid_sents_posids=test_text2ids[3],
                    valid_sents_caseids=test_text2ids[4]
                    )

with open('Result/Data/MalwareDB/test_id_dict.pickle', 'wb') as file:
    pickle.dump(test_id_dict, file)  # can be used for others

# pad the_word_id, label_id, pos_id, and case_id of train_sentence
train_word_sentence_padding = sentence_padding(train_text2ids[0], sent_maxlen, build_vocab_result[0]['[PAD]'])
# train_word_sentence_padding = torch.tensor(train_word_sentence_padding, dtype=torch.long)
train_label_sentence_padding = sentence_padding(train_text2ids[2], sent_maxlen, build_vocab_result[4]['[X]'])
train_pos_sentence_padding = sentence_padding(train_text2ids[3], sent_maxlen, build_vocab_result[6]['[PPAD]'])
train_case_sentence_padding = sentence_padding(train_text2ids[4], sent_maxlen, case2idx['[PAD]'])
# pad the char_id_sentence
train_char_sentences_padding = char_sentences_padding(train_text2ids[1], sent_maxlen, word_maxlen)
# train_char_sentences_padding = torch.tensor(train_char_sentences_padding, dtype=torch.long)

train_id_pad_dict = dict(train_wordids_pad=train_word_sentence_padding,
                         train_charids_pad=train_char_sentences_padding,
                         train_labelids_pad=train_label_sentence_padding,
                         train_posids_pad=train_pos_sentence_padding,
                         train_caseids_pad=train_case_sentence_padding
                         )

with open('Result/Data/MalwareDB/train_id_pad_dict.pickle', 'wb') as file:
    pickle.dump(train_id_pad_dict, file)  # can be used for train and pos_emb

# pad the_word_id, label_id, pos_id, and case_id of valid_sentence
valid_word_sentence_padding = sentence_padding(valid_text2ids[0], sent_maxlen, build_vocab_result[0]['[PAD]'])
# valid_word_sentence_padding = torch.tensor(valid_word_sentence_padding, dtype=torch.long)
valid_label_sentence_padding = sentence_padding(valid_text2ids[2], sent_maxlen, build_vocab_result[4]['[X]'])
valid_pos_sentence_padding = sentence_padding(valid_text2ids[3], sent_maxlen, build_vocab_result[6]['[PPAD]'])
valid_case_sentence_padding = sentence_padding(valid_text2ids[4], sent_maxlen, case2idx['[PAD]'])
# pad the char_id_sentence
valid_char_sentences_padding = char_sentences_padding(valid_text2ids[1], sent_maxlen, word_maxlen)
# valid_char_sentences_padding = torch.tensor(valid_char_sentences_padding, dtype=torch.long)

valid_id_pad_dict = dict(valid_wordids_pad=valid_word_sentence_padding,
                         valid_charids_pad=valid_char_sentences_padding,
                         valid_labelids_pad=valid_label_sentence_padding,
                         valid_posids_pad=valid_pos_sentence_padding,
                         valid_caseids_pad=valid_case_sentence_padding
                         )

with open('Result/Data/MalwareDB/valid_id_pad_dict.pickle', 'wb') as file:
    pickle.dump(valid_id_pad_dict, file)  # can be used for validation and pos_emb

# pad the_word_id, label_id, pos_id, and case_id of test_sentence
test_word_sentence_padding = sentence_padding(test_text2ids[0], sent_maxlen, build_vocab_result[0]['[PAD]'])
# test_word_sentence_padding = torch.tensor(test_word_sentence_padding, dtype=torch.long)
test_label_sentence_padding = sentence_padding(test_text2ids[2], sent_maxlen, build_vocab_result[4]['[X]'])
test_pos_sentence_padding = sentence_padding(test_text2ids[3], sent_maxlen, build_vocab_result[6]['[PPAD]'])
test_case_sentence_padding = sentence_padding(test_text2ids[4], sent_maxlen, case2idx['[PAD]'])
# pad the char_id_sentence
test_char_sentences_padding = char_sentences_padding(test_text2ids[1], sent_maxlen, word_maxlen)
# test_char_sentences_padding = torch.tensor(test_char_sentences_padding, dtype=torch.long)

test_id_pad_dict = dict(test_wordids_pad=test_word_sentence_padding,
                        test_charids_pad=test_char_sentences_padding,
                        test_labelids_pad=test_label_sentence_padding,
                        test_posids_pad=test_pos_sentence_padding,
                        test_caseids_pad=test_case_sentence_padding
                        )

with open('Result/Data/MalwareDB/test_id_pad_dict.pickle', 'wb') as file:
    pickle.dump(test_id_pad_dict, file)  # can be used for test and pos_emb

# get all the feature_tables
np.save('Result/Embedding/MalwareDB/case_embedding.npy', case_emb)
# case_emb_table = torch.tensor(case_emb, dtype=torch.float32)
# pos_emb_table = torch.tensor(build_pos_emb_table(), dtype=torch.float32)
np.save('Result/Embedding/MalwareDB/char_embedding.npy', build_char_emb_table(build_vocab_result[3]))
# char_emb_table = torch.tensor(build_char_emb_table(build_vocab_result[3]), dtype=torch.float32)

glove = GloveFeature('Result/Embedding/glove.6B.50d.txt') # 该地址后续可能会变
glove_embedding_dict = glove.load_glove_embedding()
word_emb_table = build_word_emb_table(build_vocab_result[1], glove_embedding_dict, glove.glove_dim)
np.save('Result/Embedding/MalwareDB/word_embedding.npy', word_emb_table)
# word_emb_table = torch.tensor(word_emb_table, dtype=torch.float32)
