#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 下午2:57
# @Author  : PeiP Liu
# @FileName: Bert_data_utils.py
# @Software: PyCharm

import torch
from torch.utils.data import Dataset
# from keras_preprocessing.sequence import pad_sequences


class InputFeature():
    def __init__(self, input_ids, input_mask, seg_ids, first_label_mask, true_label_ids, true_label_mask):
    # def __init__(self, input_ids, input_mask, seg_ids, word_token_num, true_label_ids, true_label_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.seg_ids = seg_ids
        self.first_label_mask = first_label_mask
        # self.word_token_num = word_token_num
        self.true_label_ids = true_label_ids
        # the true_label_mask can also be got by pad_sequences(word_token_num.bool().int(),...)
        self.true_label_mask = true_label_mask


class DataProcessor():
    def __init__(self, sentences, sentence_labels, tokenizer, max_seq_len, label2idx):
        self.sentences = sentences
        self.sentence_labels = sentence_labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # the top 3 tags in label_list are '[BOS]' '[EOS]' 'X'
        self.label2idx = label2idx
        # self.label_num = len(label2idx)

    def sentence2feature(self, sentence, sentence_label):

        tokens = ['[CLS]']  # the beginning of a sentence
        first_label_mask = [0]  # when we get the valid tag, padding's tag should be ignored
        true_label_ids = []
        # word_token_num = []

        for i, i_word in enumerate(sentence):  # the sentence is original text, which not contains pad, bos and eos
            sub_tokens = self.tokenizer.tokenize(i_word)
            if not sub_tokens:
                sub_tokens = ['[UNK]']
            tokens.extend(sub_tokens)
            true_label_ids.append(self.label2idx[sentence_label[i]])
            # word_token_num.append(len(sub_tokens))
            for j in range(len(sub_tokens)):
                if j == 0:
                    first_label_mask.append(1)
                    # we only record the first tag of sub_tokens, simpler
                else:
                    first_label_mask.append(0)

        # truncating before filling
        if len(tokens) > self.max_seq_len-1:
            tokens = tokens[:self.max_seq_len-1]
            first_label_mask = first_label_mask[:self.max_seq_len-1]

        # filling
        tokens = tokens + ['[SEP]']
        first_label_mask.append(0)
        input_mask = len(tokens) * [1]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(input_ids) < self.max_seq_len:
            first_label_mask.append(0)
            input_mask.append(0)
            input_ids.append(0)
            # we also can use the following for padding
            # input_ids = pad_sequences(input_ids, maxlen=self.max_seq_len, dtype='long', value=0,
            # truncating='post', padding='post')

        if len(true_label_ids) > self.max_seq_len:
            true_label_ids = true_label_ids[:self.max_seq_len]
            # word_token_num = word_token_num[:self.max_seq_len]

        true_label_mask = len(true_label_ids) * [1]

        while len(true_label_ids) < self.max_seq_len:
            true_label_ids.append(0)
            true_label_mask.append(0)

        seg_ids = self.max_seq_len * [0]

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(first_label_mask) == self.max_seq_len
        assert len(seg_ids) == self.max_seq_len
        assert len(true_label_ids) == self.max_seq_len
        assert len(true_label_mask) == self.max_seq_len

        return input_ids, input_mask, seg_ids, first_label_mask, true_label_ids, true_label_mask
        # return input_ids, input_mask, seg_ids, word_token_num, true_label_ids, true_label_mask

    def get_features(self):
        features = []
        for sentence, sentence_label in zip(self.sentences, self.sentence_labels):
            ii, im, si, flm, tli, tlm = self.sentence2feature(sentence, sentence_label)
            features.append(InputFeature(ii, im, si, flm, tli, tlm))
        return features


class BertCRFData(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        # every iteration will return a tuple, which contains the following:
        return feature.input_ids, feature.input_mask, feature.seg_ids, feature.first_label_mask, \
               feature.true_label_ids, feature.true_label_mask
        # return feature.input_ids, feature.input_mask, feature.seg_ids, feature.word_token_num, \
        #                feature.true_label_ids, feature.true_label_mask

    @classmethod
    def seq_tensor(cls, batch):  # the batch are results of batch_size __getitem__()
        # we also can use it for padding
        # padding = lambda x, max_seq_len: [feature[x]+(max_seq_len-len(feature[x]))*[0] for feature in batch]
        # input_ids = torch.tensor(padding(0, max_seq_len))

        list2tensor = lambda x: torch.tensor([feature[x] for feature in batch], dtype=torch.long)
        input_ids = list2tensor(0)
        input_mask = list2tensor(1)
        seg_ids = list2tensor(2)
        first_label_mask = list2tensor(3)
        # word_token_num = list2tensor(3)
        true_label_ids = list2tensor(4)
        true_label_mask = list2tensor(5)
        return input_ids, input_mask, seg_ids, first_label_mask, true_label_ids, true_label_mask
        # return input_ids, input_mask, seg_ids, word_token_num, true_label_ids, true_label_mask
