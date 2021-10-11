#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/6 上午9:27
# @Author  : PeiP Liu
# @FileName: model.py
# @Software: PyCharm


import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F

import sys
sys.path.append("..")
from BiLSTM_CNN.char_CNN import CNN
from common_modules.Transformer_module import MultiHeadAttention, FeedForward
from common_modules.CRF_classes import CRF
from common_modules.Attention import build_attention
from common_modules.utils import decode_tag

from BiLSTM_CNN.security_augmentation import SoftInternalAugmentation, SoftAugmentationAttention, HardInternalAugmentation

Model_Registry = {}


def build_model(model_name, *args, **kwargs):
    return Model_Registry[model_name](*args, **kwargs)  # 模型初始化


def register_model(name):
    def register_model_cls(cls):
        if name in Model_Registry:
            raise ValueError("There had been the model!")
        if not issubclass(cls, BaseSetting):
            raise ValueError("The model must extend BaseModel")
        Model_Registry[name] = cls  # 模型类字典｛类名：类｝
        return cls
    return register_model_cls


class BaseSetting(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.word_emb_table = torch.tensor(args.word_emb_table, dtype=torch.float32)
        self.char_emb_table = torch.tensor(args.char_emb_table, dtype=torch.float32)
        self.pos_emb_table = torch.tensor(args.pos_emb_table, dtype=torch.float32)
        self.case_emb_table = torch.tensor(args.case_emb_table, dtype=torch.float32)

        self.sent_maxlen = args.real_sent_maxlen
        self.word_maxlen = args.real_word_maxlen

        self.word_pad_indx = args.word_pad_indx

        self.device = args.device

        # self.char_embed = nn.Embedding.from_pretrained(self.char_emb_table, freeze=True)
        self.char_embed = nn.Embedding(self.char_emb_table.shape[0], self.char_emb_table.shape[1])
        nn.init.xavier_normal_(self.char_embed.weight)
        self.word_embed = nn.Embedding.from_pretrained(self.word_emb_table, freeze=False)
        # self.word_embed = nn.Embedding(self.word_emb_table.shape[0], self.word_emb_table.shape[1])
        nn.init.xavier_normal_(self.word_embed.weight)
        self.pos_embed = nn.Embedding.from_pretrained(self.pos_emb_table, freeze=True)
        # self.pos_embed = nn.Embedding(self.pos_emb_table.shape[0], self.pos_emb_table.shape[1])
        nn.init.xavier_normal_(self.pos_embed.weight)
        self.case_embed = nn.Embedding.from_pretrained(self.case_emb_table, freeze=True)
        # self.case_embed = nn.Embedding(self.case_emb_table.shape[0], self.case_emb_table.shape[1])
        # nn.init.xavier_normal_(self.case_embed.weight)

        self.num_labels = args.num_labels
        self.label_pad_indx = args.label_pad_indx
        self.label_bos_indx = args.label_bos_indx
        self.label_eos_indx = args.label_eos_indx

        self.transformer_num_blocks = args.transformer_num_blocks
        self.transformer_num_heads = args.transformer_num_heads

        self.bilstm_layers = args.bilstm_layers

        self.input_dim = args.input_dim
        self.hid_dim = args.hid_dim
        self.model_dim = args.model_dim

        self.lstmcnn_sigmoid = nn.Sigmoid()
        self.lstmcnn_tanh = nn.Tanh()
        self.lstmcnn_relu = nn.ReLU()
        self.lstmcnn_leaky = nn.LeakyReLU(0.2)
        self.input_norm = nn.BatchNorm1d(self.input_dim)

        self.dropout_rate = args.dropout_rate
        self.attention_type = args.attention_type

        self.input2mod = nn.Linear(self.input_dim, self.model_dim)
        # 中间包括各种特征变换的方式，如attention、lstm
        self.mod2emission = nn.Linear(self.model_dim, self.num_labels)
        self.hid2emission = nn.Linear(self.hid_dim, self.num_labels)

        # create the gate of inaug
        self.inaug_gate = nn.Parameter(torch.empty(2 * self.hid_dim, self.hid_dim), requires_grad=True)
        nn.init.xavier_normal_(self.inaug_gate)
        self.internal_aug_fusion = nn.Linear(self.hid_dim, self.model_dim)

        # cteare the gate of outaug
        self.outaug_gate = nn.Parameter(torch.empty(2 * self.num_labels, self.num_labels), requires_grad=True)
        nn.init.xavier_normal_(self.outaug_gate)
        self.bert_aug_fusion = nn.Linear(self.num_labels, self.num_labels)
        
        self.numlabels_2_numlabels = nn.Linear(2*self.num_labels, self.num_labels)
        self.hiddim_2_hiddim = nn.Linear(2*self.hid_dim, self.hid_dim)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.cnn_model = CNN(self.char_emb_table.size(-1), self.word_maxlen)

        self.SIA = SoftInternalAugmentation(args)
        self.SAA = SoftAugmentationAttention(args.word2vec, args.hid_dim, args.dropout_rate)

        self.HIA = HardInternalAugmentation(args)
        self.hard_augmented_emb_table = self.HIA.hard_augmentation_embedding(self.hid_dim)

        self.crf = CRF(
            num_labels=self.num_labels,
            pad_idx=self.label_pad_indx,
            bos_idx=self.label_bos_indx,
            eos_idx=self.label_eos_indx,
            device=args.device
        )

    """
    def init_embedding(self, pre_trained_weight):  # 这里还需要加入别的权值向量，比如pos_embed、word_embed
        # rf https://github.com/liu-nlper/SLTK/blob/master/sltk/nn/modules/feature.py
        new_state_dict = {'word_embed.weight': pre_trained_weight}
        state_dict = self.state_dict()
        state_dict.update(new_state_dict)
        self.load_state_dict(state_dict)
        for para_name, tensor in self.named_parameters():
            if para_name == 'word_embed.weight':
                tensor.requires_grad = False

    def ano_init_embedding(self, case_emb_table):
        self.case_embed.weight.data.copy_(torch.from_numpy(case_emb_table)) # 此处的case_emb_table应该是numpy.array()类型
        self.case_embed.weight.requires_grad = False
    """

    def _char_cnn(self, batch_char_data):
        assert self.sent_maxlen == batch_char_data.size(1)  # 保证我们的数据shape形式是(batch_size, sent_len, word_len)
        assert self.word_maxlen == batch_char_data.size(2)
        char_data_emb = self.char_embed(batch_char_data)   # (batch_size, sent_maxlen, word_maxlen, char_emb_dim)
        char_data_emb = char_data_emb.permute(0, 3, 1, 2)  # (batch_size, char_emb_dim, sent_maxlen, word_maxlen)
        """
        # we can also use the conv3d, please rf https://github.com/liu-nlper/SLTK/blob/master/sltk/nn/modules/feature.py
        batch_char_data = batch_char_data.view(-1, self.sent_maxlen * self.word_maxlen) # (batch_size, -1)
        char_data_emb = self.char_emb_table[batch_char_data] # (batch_size, -1, char_emb_dim)
        char_data_emb = char_data_emb.view(-1, 1, self.sent_maxlen, self.word_maxlen, self.char_emb_dim)
        f = nn.Conv3d(in_channels = 1,out_channels = filter_num, kernel_size = (1, filter_size, self.char_emb_dim))
        f = f.cuda()
        cnn_feature = f(char_data_emb)
        """

        """
        # another conv2d()
        char_data_emb = char_data_emb.view(batch_size * sent_len, 1, word_len, in_char_dim).permute(0, 3, 1, 2)
        conv = nn.conv2d(in_char_dim, out_char_dim, filter = (1, filter_size))
        conv_result = conv(char_data_emb) # (batch*sent_len, out_char_dim, 1, word_len-file_size+1)
        conv_result = conv_result.view(batch*sent_len, out_char_dim, word_len-file_size+1)
        """
        cnn_feature = self.cnn_model(char_data_emb)  # the input size should be (batch_size, in_char_dim, sent_len, word_len)
        return cnn_feature  # the size is (batch_size, sent_len, out_char_dim)

    """
    def forward(self, char_data, word_data, pos_data, case_data):
        # 这里以q的sent中每个元素去检测k的sent所有元素的相似性而构造的batch*sent_len*sent_len的mask。mask中padding位置为真，方便以后的计算。
        # (batch_size, q_sent_len, k_sent_len)
        self.attention_mask = word_data.eq(self.word_pad_indx).unsqueeze(1).expand(-1, word_data.size(-1), -1)

        self.char_feature = self._char_cnn(batch_char_data=char_data)
        self.word_feature = self.word_embed[word_data]
        self.pos_feature = self.pos_embed[pos_data]
        self.case_feature = self.case_embed[case_data]

        input_fusion_embedding = torch.cat([self.word_feature, self.char_feature, self.pos_feature, self.case_feature], dim=-1) # 多特征融合

        self.hidden_feature = self.input2mod(input_fusion_embedding)

        self.valid_sent_mask = (word_data != self.word_pad_indx)
        # or we can use:
        # valid_sent_mask = (truth_label != self.label_pad_indx)
    """

    def partial_forward(self, char_data, word_data, pos_data, case_data):
        # (batch_size, sent_len, sent_len)
        attention_mask = word_data.eq(self.word_pad_indx).unsqueeze(1).expand(-1, word_data.size(-1), -1)

        char_feature = self._char_cnn(batch_char_data=char_data)
        word_feature = self.word_embed(word_data)
        pos_feature = self.pos_embed(pos_data)
        case_feature = self.case_embed(case_data)
        
        # print(char_feature.size(-1))
        # print(word_feature.size(-1))
        # print(pos_feature.size(-1))
        # print(case_feature.size(-1))

        # 多特征融合，(batch_size, sent_len, input_dim)
        input_fusion_embedding = torch.cat([word_feature, char_feature, pos_feature, case_feature], dim=-1)
        input_fusion_embedding = self.dropout(input_fusion_embedding)
        input_fusion_embedding = self.input_norm(input_fusion_embedding.permute(0, 2, 1).contiguous())

        mod_feature = self.input2mod(input_fusion_embedding.permute(0, 2, 1).contiguous())  # (batch_size, sent_len, model_dim)

        valid_sent_mask = (word_data != self.word_pad_indx)  # (batch_size, sent_len)
        # or we can use:
        # valid_sent_mask = (truth_label != self.label_pad_indx)

        return attention_mask, mod_feature, valid_sent_mask

    def init_lstm_hidden(self, batch_size, num_directs):
        # h_0_shape = c_0_shape = (num_layers*num_directs, batch_size, hidden_size), abide the orig requirement
        return (torch.randn(self.bilstm_layers*num_directs, batch_size, self.hid_dim//2).to(self.device),
               torch.randn(self.bilstm_layers*num_directs, batch_size, self.hid_dim//2).to(self.device))

"""
@register_model('transformer_crf')
class TansformerCRF(BaseSetting):
    def __init__(self, args):
        super().__init__(args)  # https://blog.csdn.net/u012308586/article/details/109488908
        '''
        self.case_embed = nn.Embedding(self.case_emb_table.size(0), self.case_emb_table.size(-1))
        self.word_embed = nn.Embedding(self.word_emb_table.size(0), self.word_emb_table.size(-1))
        '''

        for i_block in range(self.transformer_num_blocks):
            self.__setattr__('MultiHeadAttention_{}'.format(i_block), MultiHeadAttention(
                model_dim=self.model_dim,
                num_head=self.transformer_num_heads,
                dropout_rate=self.dropout_rate,
                attention_type=self.attention_type
            ))
            self.__setattr__('FeedForward_{}'.format(i_block), FeedForward(
                model_dim=self.model_dim,
                hidden_dim=self.hid_dim,
                dropout_rate=self.dropout_rate
            ))

    def forward(self, char_data, word_data, pos_data, case_data, truth_label, opt_type):
        super().forward(char_data, word_data, pos_data, case_data)

        for i_block in range(self.transformer_num_blocks):
            self.hidden_feature, _ = self.__getattr__('MultiHeadAttention_{}'.format(i_block))(self.hidden_feature,
                                                                      self.hidden_feature, self.hidden_feature, attention_mask=self.attention_mask)
            self.hidden_feature = self.__getattr__('FeedForward_{}'.format(i_block))(self.hidden_feature)

        feature2emission = self.mod2emission(self.hidden_feature)  # (batch_size, sent_len, num_labels)

        if opt_type is 'train':
            loss = self.crf(feature2emission, truth_label, self.valid_sent_mask)
            return loss, 0

        elif opt_type is "test":
            final_max_score, best_tagging_path = self.crf.viterbi_decode(feature2emission, self.valid_sent_mask)
            return final_max_score, best_tagging_path

        else:
            print("Please input the correct string, 'train' or 'test'")
            raise ValueError
"""


@register_model('transformer_crf')
class TansformerCRF(BaseSetting):
    def __init__(self, args):
        super().__init__(args)  # https://blog.csdn.net/u012308586/article/details/109488908

        for i_block in range(self.transformer_num_blocks):
            self.__setattr__('MultiHeadAttention_{}'.format(i_block), MultiHeadAttention(
                model_dim=self.model_dim,
                num_head=self.transformer_num_heads,
                dropout_rate=self.dropout_rate,
                attention_type=self.attention_type
            ))
            self.__setattr__('FeedForward_{}'.format(i_block), FeedForward(
                model_dim=self.model_dim,
                hidden_dim=self.hid_dim,
                dropout_rate=self.dropout_rate
            ))

    def forward(self, char_data, word_data, pos_data, case_data, truth_label, opt_type, bert_emission):
        attention_mask, mod_feature, valid_sent_mask = self.partial_forward(char_data, word_data, pos_data, case_data)

        for i_block in range(self.transformer_num_blocks):
            # input_shape = output_shape = (batch_size, sent_len, self.model_dim)
            mod_feature, _ = self.__getattr__('MultiHeadAttention_{}'.format(i_block))(mod_feature,
                                                                      mod_feature, mod_feature, attention_mask)
            # input_shape = output_shape = (batch_size, sent_len, self.model_dim)
            mod_feature = self.__getattr__('FeedForward_{}'.format(i_block))(mod_feature)

        # input_shape =(batch_size, sent_len, self.model_dim), output_shape =  (batch_size, sent_len, num_labels)
        emission = self.mod2emission(mod_feature)
        emission = self.lstmcnn_tanh(emission)
        emission = self.dropout(emission)

        """
        if opt_type is 'train':
            loss = self.crf(emission, truth_label, valid_sent_mask)
            return loss

        elif opt_type is "test":
            final_max_score, best_tagging_path = self.crf.viterbi_decode(emission, valid_sent_mask)
            return final_max_score, best_tagging_path

        else:
            print("Please input the correct string, 'train' or 'test'")
            raise ValueError
        """
        if opt_type is 'train':
            loss = F.cross_entropy(emission.view(-1, self.num_labels), truth_label.view(-1), ignore_index=self.label_pad_indx)
            return loss

        elif opt_type is "test":
            pre_tagging_path = decode_tag(emission, valid_sent_mask)
            return pre_tagging_path

        else:
            print("Please input the correct string, 'train' or 'test'")
            raise ValueError


@register_model('bilstm_atten_crf')
class BilstmAttenCRF(BaseSetting):
    def __init__(self, args):
        super(BilstmAttenCRF, self).__init__(args)
        # nn.LSTM(input_size, hidden_size, num_layers)
        # so here input_size = self.model_dim, hidden_size = self.hid_dim//2
        self.bilstm = nn.LSTM(self.model_dim, self.hid_dim//2,
                              num_layers=self.bilstm_layers, batch_first=True, bidirectional=True)

        self.attention = build_attention(self.attention_type, self.hid_dim, self.hid_dim, self.dropout_rate)  # attention_type=general

    def forward(self, char_data, word_data, pos_data, case_data, truth_label, opt_type, bert_emission):
        attention_mask, mod_feature, valid_sent_mask = self.partial_forward(char_data, word_data, pos_data, case_data)

        # if batch_first=False (else, we should transpose(0,1)):
        # output, (h_n, c_n) = self.bilstm(input, (h_0, c_0))
        # input_shape = (sent_len, batch_size, input_size)
        # h_0_shape = c_0_shape = (num_layers*num_directs, batch_size, hidden_size)
        # output_shape = (sent_len, batch_size, num_directs*hidden_size)
        # h_n_shape = c_n_shape = (num_layers*num_directs, batch_size, hidden_size)

        # input_shape=(batch_size, sent_len, self.model_dim), lstm_output_shape=(batch_size, sent_len, self.hid_dim)
        lstm_output, _ = self.bilstm(mod_feature, self.init_lstm_hidden(word_data.size(0), 2))
        lstm_output = self.lstmcnn_tanh(lstm_output)

        # *******************在此阶段添加外部语义增强信息,并进行ff*******************

        # the shape of attention_output is (batch_size, sent_len, self.hid_dim)
        attention_output, _ = self.attention(lstm_output, lstm_output, lstm_output, attention_mask)

        emission = self.hid2emission(attention_output)
        emission = self.lstmcnn_relu(emission)
        emission = self.dropout(emission)

        # *******************在此阶段添加BERT信息，并进行ff*******************

        """
        if opt_type is 'train':
            loss = self.crf(emission, truth_label, valid_sent_mask)
            return loss

        elif opt_type is "test":
            final_max_score, best_tagging_path = self.crf.viterbi_decode(emission, valid_sent_mask)
            return final_max_score, best_tagging_path

        else:
            print("Please input the correct string, 'train' or 'test'")
            raise ValueError
        """
        if opt_type is 'train':
            loss = F.cross_entropy(emission.view(-1, self.num_labels), truth_label.view(-1), ignore_index=self.label_pad_indx)
            return loss

        elif opt_type is "test":
            pre_tagging_path = decode_tag(emission, valid_sent_mask)
            return pre_tagging_path

        else:
            print("Please input the correct string, 'train' or 'test'")
            raise ValueError


@register_model('bilstm_multihead_atten_crf')
class BilstmMultiheadAttenCRF(BaseSetting):
    def __init__(self, args):
        super(BilstmMultiheadAttenCRF, self).__init__(args)
        self.bilstm = nn.LSTM(self.model_dim, self.hid_dim//2,
                              num_layers=self.bilstm_layers, batch_first=True, bidirectional=True)

        self.multihead_attention = MultiHeadAttention(
                model_dim=self.hid_dim,
                num_head=self.transformer_num_heads,
                dropout_rate=self.dropout_rate,
                # attention_type=self.attention_type)
                )
        self.FF = FeedForward(model_dim=self.hid_dim, hidden_dim=self.hid_dim, dropout_rate=self.dropout_rate)

    def forward(self, char_data, word_data, pos_data, case_data, truth_label, opt_type, bert_emission):
        # mod_feature_shape=(batch_size, sent_len, self.model_dim)
        attention_mask, mod_feature, valid_sent_mask = self.partial_forward(char_data, word_data, pos_data, case_data)

        # input_shape=(batch_size, sent_len, self.model_dim), lstm_output_shape=(batch_size, sent_len, self.hid_dim)
        lstm_output, _ = self.bilstm(mod_feature, self.init_lstm_hidden(word_data.size(0), 2))
        lstm_output = self.lstmcnn_tanh(lstm_output)
        lstm_output = self.dropout(lstm_output)

        # *******************augmented semantics from internal corpus*******************
        # hard augmentation
        # (batch_size, sent_len, self.hid_dim)
        hard_augmented_feature = self.hard_augmented_emb_table[word_data]
        # (batch_size, sent_len, 2*self.hid_dim)
        cat_feature = torch.cat([lstm_output, hard_augmented_feature], dim=-1)
        # in_gate = cat_feature.matmul(self.inaug_gate)
        in_gate = self.hiddim_2_hiddim(cat_feature)

        # option
        # soft augmentation
        # soft_aug_words = self.SIA.soft_augmentation_words(word_data)  # (batch_size, sent_len, sim_num)
        # soft_augmented_feature = self.SAA(lstm_output, soft_aug_words)  # (batch_size, sent_len, hid_dim)
        # (batch_size, sent_len, 2*self.hid_dim)
        # in_cat_feature = torch.cat([lstm_output, soft_augmented_feature], dim=-1)
        # in_gate = in_cat_feature.matmul(self.inaug_gate)
         
        # *****************************add the gate module here*****************************
        in_gate = self.lstmcnn_sigmoid(in_gate)
        inaug_one = torch.ones(in_gate.shape).to(self.device)
        new_mode_feature = in_gate.mul(lstm_output)+(inaug_one-in_gate).mul(hard_augmented_feature)
        # new_mode_feature = in_gate.mul(lstm_output)+(inaug_one-in_gate).mul(soft_augmented_feature)
        # new_mode_feature = self.internal_aug_fusion(new_mode_feature)
        # inaug_emission = self.mod2emission(new_mode_feature)
        inaug_emission = self.hid2emission(new_mode_feature)
        inaug_emission = self.dropout(inaug_emission)

        # *******************encoding　context after internal augmented semantics*******************
        # lstm_output, _ = self.bilstm(new_mode_feature, self.init_lstm_hidden(word_data.size(0), 2))  # unvalid, so we can ignore here.

        # lstm_output = new_mode_feature
        # lstm_output = self.hiddim_2_hiddim(torch.cat([lstm_output, new_mode_feature], dim=-1))
        # input_shape = output_shape = (batch_size, sent_len, self.hid_dim)
        attention_output, _ = self.multihead_attention(lstm_output, lstm_output, lstm_output, attention_mask)

        attention_output = self.FF(attention_output)

        emission = self.hid2emission(attention_output)
        emission = self.lstmcnn_relu(emission)
        emission = self.dropout(emission)

        # *******************augmented semantics from external corpus*******************
        bert_emission = bert_emission[:, :word_data.size(-1), :]  # bert_emission的sent_len要实际大于lstm的emission
        out_fusion_emission = torch.cat([emission, bert_emission], dim=-1)
        out_gate = out_fusion_emission.matmul(self.outaug_gate)
        out_gate = self.lstmcnn_sigmoid(out_gate)
        outaug_one = torch.ones(out_gate.shape).to(self.device)
        emission = out_gate.mul(emission)+(outaug_one-out_gate).mul(bert_emission)

        emission = self.bert_aug_fusion(emission)
        
        emission = emission + inaug_emission
        
        # emission = inaug_emission
        """
        if opt_type is 'train':
            loss = self.crf(emission, truth_label, valid_sent_mask)
            return loss

        elif opt_type is "test":
            final_max_score, best_tagging_path = self.crf.viterbi_decode(emission, valid_sent_mask)
            return final_max_score, best_tagging_path

        else:
            print("Please input the correct string, 'train' or 'test'")
            raise ValueError
        """
        if opt_type is 'train':
            loss = F.cross_entropy(emission.view(-1, self.num_labels), truth_label.view(-1), ignore_index=self.label_pad_indx)
            return loss

        elif opt_type is "test":
            pre_tagging_path = decode_tag(emission, valid_sent_mask)
            return pre_tagging_path

        else:
            print("Please input the correct string, 'train' or 'test'")
            raise ValueError


@register_model('bilstm_atten_softmax')
class BilstmAttenSoft(BaseSetting):
    def __init__(self, args):
        super(BilstmAttenSoft, self).__init__(args)
        self.bilstm = nn.LSTM(self.model_dim, self.hid_dim // 2,
                              num_layers=self.bilstm_layers, batch_first=True, bidirectional=True)

        # self.attention = build_attention(self.attention_type, self.hid_dim, self.hid_dim, self.dropout_rate)  # attention_type=general
        self.attention = build_attention('dot', dropout_rate=self.dropout_rate)  

    def forward(self, char_data, word_data, pos_data, case_data, truth_label, opt_type, bert_emission):
        attention_mask, mod_feature, valid_sent_mask = self.partial_forward(char_data, word_data, pos_data, case_data)

        lstm_output, _ = self.bilstm(mod_feature, self.init_lstm_hidden(word_data.size(0), 2))
        lstm_output = self.lstmcnn_tanh(lstm_output)
        lstm_output = self.dropout(lstm_output)
        attention_output, _ = self.attention(lstm_output, lstm_output, lstm_output, attention_mask)
        attention_output = self.dropout(attention_output)
        emission = self.hid2emission(attention_output)
        emission = self.lstmcnn_relu(emission)

        if opt_type is 'train':
            loss = F.cross_entropy(emission.view(-1, self.num_labels), truth_label.view(-1), ignore_index=self.label_pad_indx)
            return loss

        elif opt_type is "test":
            pre_tagging_path = decode_tag(emission, valid_sent_mask)
            return pre_tagging_path

        else:
            print("Please input the correct string, 'train' or 'test'")
            raise ValueError


@register_model('multi_feature_bilstm_atten_crf')
class MultifeatureBilstmAttenCRF(BaseSetting):
    def __init__(self, args):
        super(MultifeatureBilstmAttenCRF, self).__init__(args)
        self.bilstm = nn.LSTM(self.model_dim, self.hid_dim // 2,
                              num_layers=self.bilstm_layers, batch_first=True, bidirectional=True)

        # self.attention = build_attention(self.attention_type, self.hid_dim, self.hid_dim, self.dropout_rate)  # attention_type=general
        self.attention = build_attention('scaled_dot', dropout_rate=self.dropout_rate)  

    def forward(self, char_data, word_data, pos_data, case_data, truth_label, opt_type, bert_emission):
        attention_mask, mod_feature, valid_sent_mask = self.partial_forward(char_data, word_data, pos_data, case_data)

        # input_shape=(batch_size, sent_len, self.model_dim), lstm_output_shape=(batch_size, sent_len, self.hid_dim)
        lstm_output, _ = self.bilstm(mod_feature, self.init_lstm_hidden(word_data.size(0), 2))
        lstm_output = self.lstmcnn_tanh(lstm_output)
        lstm_output = self.dropout(lstm_output)

        # *******************augmented semantics from internal corpus*******************
        # hard augmentation
        # (batch_size, sent_len, self.hid_dim)
        # hard_augmented_feature = self.hard_augmented_emb_table[word_data]
        # (batch_size, sent_len, self.mod_dim)
        # new_mode_feature = self.internal_aug_fusion(torch.cat([lstm_output, hard_augmented_feature], dim=-1))

        
        # option
        # soft augmentation
        soft_aug_words = self.SIA.soft_augmentation_words(word_data)  # (batch_size, sent_len, sim_num)
        soft_augmented_feature = self.SAA(lstm_output, soft_aug_words)  # (batch_size, sent_len, hid_dim)
        # (batch_size, sent_len, self.mod_dim)
        new_mode_feature = self.internal_aug_fusion(torch.cat([lstm_output, soft_augmented_feature], dim=-1))
        
        new_mode_feature = self.dropout(new_mode_feature)

        # *******************encoding　context after internal augmented semantics*******************
        lstm_output, _ = self.bilstm(new_mode_feature, self.init_lstm_hidden(word_data.size(0), 2))

        # self attention to get the most related information
        attention_output, _ = self.attention(lstm_output, lstm_output, lstm_output, attention_mask)
        attention_output = self.lstmcnn_tanh(attention_output)
        attention_output = self.dropout(attention_output)

        emission = self.hid2emission(attention_output)

        # *******************augmented semantics from external corpus*******************
        # bert_emission = bert_emission[:, :word_data.size(-1), :]  # bert_emission的sent_len要实际大于lstm的emission
        # emission = torch.cat([emission, bert_emission], dim=-1)
        emission = self.lstmcnn_relu(emission)
        emission = self.dropout(emission)

        # emission = self.bert_aug_fusion(emission)
        '''
        if opt_type is 'train':
            loss = self.crf(emission, truth_label, valid_sent_mask)
            return loss

        elif opt_type is "test":
            final_max_score, best_tagging_path = self.crf.viterbi_decode(emission, valid_sent_mask)
            return final_max_score, best_tagging_path

        else:
            print("Please input the correct string, 'train' or 'test'")
            raise ValueError

        '''
        if opt_type is 'train':
            loss = F.cross_entropy(emission.view(-1, self.num_labels), truth_label.view(-1), ignore_index=self.label_pad_indx)
            return loss

        elif opt_type is "test":
            pre_tagging_path = decode_tag(emission, valid_sent_mask)
            return pre_tagging_path

        else:
            print("Please input the correct string, 'train' or 'test'")
            raise ValueError
