#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/6 上午10:04
# @Author  : PeiP Liu
# @FileName: BertModel.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torch.nn import LayerNorm as BertLayerNorm
import sys
sys.path.append("..")
from common_modules.CRF_classes import CRF
import torch.nn.functional as F
from common_modules.utils import decode_tag


class BERT_CRF_NER(nn.Module):
    def __init__(self, bert_model, label2idx, hidden_size=768, batch_size=64, max_seq_len=256, device='cpu'):
        super(BERT_CRF_NER, self).__init__()
        self.bert_model = bert_model
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_label = len(label2idx)
        self.device = device
        # create the dropout layer
        self.dropout = nn.Dropout(0.5)
        self.bert_sigmod = nn.Sigmoid()
        # create the final nn layer to convert the output feature to emission
        self.hid2label = nn.Linear(self.hidden_size, self.num_label)
        # init the weight and bias of feature-emission layer
        nn.init.xavier_uniform_(self.hid2label.weight)
        nn.init.constant_(self.hid2label.bias, 0.0)
        self.crf = CRF(num_labels=self.num_label,
                       pad_idx=label2idx['[X]'],
                       bos_idx=label2idx['[BOS]'],
                       eos_idx=label2idx['[EOS]'],
                       device=device)
        # self.apply(self.init_bert_weight)

    def init_bert_weight(self, module):
        # cf https://github.com/Louis-udm/NER-BERT-CRF/blob/master/NER_BERT_CRF.py
        # rf https://www.cnblogs.com/BlueBlueSea/p/12875517.html
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_bert_features(self, input_ids, seg_ids, atten_mask):
        # rf https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        outputs = self.bert_model(input_ids, token_type_ids=seg_ids,
                                  attention_mask=atten_mask, output_hidden_states=True, output_attentions=True)
        # last_hidden_states = outputs[0]
        last_hidden_states = outputs.last_hidden_state # (batch_size, seq_length, hidden_size)

        # pooler_outputs = outputs[1]
        # the feature of [CLS], and it represents the feature of whole sentence
        # We can better average or pool the sequence of hidden-states for the whole sequence.
        pooler_outputs = outputs.pooler_output # (batch_size, hidden_size)

        # all_hidden_states = outputs[2]
        all_hidden_states = outputs.hidden_states # (1+12, batch_size, seq_length, hidden_size)
        # one for the output of the embeddings(1), and the others for the output of each hidden layer(12)
        hidden_num = len(all_hidden_states) # 13
        embedding_output = all_hidden_states[0] # (batch_size, seq_length, hidden_size)
        attention_hidden_states = all_hidden_states[1:] # (12, batch_size, seq_length, hidden_size)
        # reciprocal i_th layer_states, and the j_th token feature at the layer
        # rec_i_layer_states = all_hidden_states[-i] # (batch_size, seq_length, hidden_size)
        # rec_i_layer_j_token_states = rec_i_layer_states[:,j] # (batch_size, hidden_size)

        # all_attentions = outputs[3]
        all_attentions = outputs.attentions
        # attention_num = len(all_attentions) # 12, one less than hidden_num because of the embedding layer

        return last_hidden_states

    def word_embedding(self, token_features, input_mask, first_label_mask):
        batch_size, seq_len, feature_dim = token_features.shape
        wv = torch.zeros(token_features.shape, dtype=torch.float32).to(self.device)
        for batch_iter in range(batch_size):
            # get the valid information except for [CLS] and [SEP]
            valid_token_input = token_features[batch_iter][input_mask[batch_iter].bool()][1:-1]
            valid_label_mask = first_label_mask[batch_iter][input_mask[batch_iter].bool()][1:-1]
            # we also can use the following to achieve the result
            # valid_token_input = torch.masked_select(token_features[batch_iter],
            #                                         input_mask[batch_iter].unsqueeze(1)).view(-1,feature_dim)[1:-1]
            # valid_label_mask = torch.masked_select(first_label_mask[batch_iter],
            #                                        input_mask[batch_iter].unsqueeze(1)).view(-1,feature_dim)[1:-1]
            assert len(valid_token_input) == len(valid_label_mask)
            # print(valid_label_mask)
            assert valid_label_mask[0].item() == 1  # transfer tensor to num
            i_word_vector = 0
            token_iter = 0
            while token_iter < len(valid_label_mask):
                token_post = token_iter+1
                while token_post < len(valid_label_mask) and valid_label_mask[token_post] != 1:
                    token_post = token_post + 1
                # we make the token embedding summing for word_vector
                # word_vector = valid_token_input[token_iter:token_post].sum(0)
                # or, we can use the averaging for word vector
                word_vector = valid_token_input[token_iter:token_post].mean(0)
                wv[batch_iter][i_word_vector] = word_vector
                i_word_vector = i_word_vector + 1
                token_iter = token_post
        # add dropout onto feature
        dropout_feature = self.dropout(wv)
        # convert the output feature from bert to label distribution
        wv2emission = self.hid2label(dropout_feature)
        # wv2emission = torch.relu(wv2emission)
        return wv2emission

    """
    def word_embedding2(self, token_features, input_mask, word_token_num, true_label_ids, true_label_mask):
        batch_size, seq_len, feature_dim = token_features.shape
        wv = torch.zeros(token_features.shape, dtype=torch.float32).to(self.device)
        for batch_iter in range(batch_size):
            assert len(word_token_num[batch_iter]) == len(true_label_ids[batch_iter][true_label_mask[batch_iter].bool()])
            # get the valid information except for [CLS] and [SEP]
            valid_token_input = token_features[batch_iter][input_mask[batch_iter]][1:-1]
            token_num = word_token_num[batch_iter]
            assert len(valid_token_input) == word_token_num[batch_iter].sum()
            i_word_vector = 0
            token_iter = 0
            while token_iter < len(valid_token_input) and i_word_vector < len(token_num):
                token_post = token_iter + token_num[i_word_vector]
                # we make the token embedding summing for word_vector
                word_vector = valid_token_input[token_iter:token_post].sum(0)
                # or, we can use the averaging for word vector
                # word_vector = valid_token_input[token_iter:token_post].mean(0)
                wv[batch_iter][i_word_vector] = word_vector
                i_word_vector = i_word_vector + 1
                token_iter = token_post
        # add dropout onto feature
        dropout_feature = self.dropout(wv)
        # convert the output feature from bert to label distribution
        wv2emission = self.hid2label(dropout_feature)
        return wv2emission
    """

    def neg_log_likehood(self, input_ids, input_mask, seg_ids, first_label_mask, true_label_ids, true_label_mask):
    # def neg_log_likehood(self, input_ids, input_mask, seg_ids, word_token_num, true_label_ids, true_label_mask):
        token_features = self.get_bert_features(input_ids, seg_ids, input_mask)
        # we ignore the feature of [CLS] and [SEP] of bert and fuse the token feature wv based on valid_input_mask.
        wv2emission = self.word_embedding(token_features, input_mask, first_label_mask)
        wv2emission = self.bert_sigmod(wv2emission)
        # wv2emission = self.word_embedding2(token_features, input_mask, word_token_num, true_label_ids, true_label_mask)
        # object_score = self.crf(features=wv2emission, tag_seqs=true_label_ids, mask=true_label_mask)
        object_score = F.cross_entropy(wv2emission.view(-1, 10), true_label_ids.view(-1), ignore_index=2)
        return object_score

    def forward(self, input_ids, input_mask, seg_ids, first_label_mask, true_label_mask):
        token_feature = self.get_bert_features(input_ids, seg_ids, input_mask)
        word2emission = self.word_embedding(token_feature, input_mask, first_label_mask)
        word2emission = self.bert_sigmod(word2emission)
        # return self.crf.viterbi_decode(word2emission, true_label_mask)
        return decode_tag(word2emission, true_label_mask)

    def get_bert_emission(self, input_ids, input_mask, seg_ids, first_label_mask):
        token_features = self.get_bert_features(input_ids, seg_ids, input_mask)
        wv2emission = self.word_embedding(token_features, input_mask, first_label_mask)
        return wv2emission  # (batch_size, token_sent_len, bert_token_dim)

