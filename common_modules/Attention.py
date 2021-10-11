#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/6 下午8:48
# @Author  : PeiP Liu
# @FileName: Attention.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Attention_Module = {}


def build_attention(attention_type, *args, **kwargs):
    return Attention_Module[attention_type](*args, **kwargs)  # 相当于返回了类cls


def create_attention_name(name):

    def create_attention_class(cls):
        if name in Attention_Module:
            raise ValueError("The attention model has been created")
        if not issubclass(cls, BaseAttention):
            raise ValueError("Attention ({}:{}) must extend BaseAttention".format(name, cls.__name__))
        Attention_Module[name] = cls
        return cls

    return create_attention_class


class BaseAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

# the q_shape = k_shape = v_shape = input_shape = output_shape = (batch_size, sent_len, emb_dim)


@create_attention_name('dot')
class DotProductAttention(BaseAttention):
    def __init__(self, dropout_rate=0.0, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, attetion_mask=None):
        attention_matrix = torch.bmm(q, k.permute(0, 2, 1).contiguous())  # (batch_size, sent_len, sent_len)

        if attetion_mask is not None:
            # rf https://blog.csdn.net/chengyq116/article/details/106961087
            # -np.inf is used to replace "true" in attetion_mask(in fact the place where ele is padded)
            # as long as earlier than softmax
            attention_matrix.masked_fill_(attetion_mask, -np.inf)  # (batch_size, q_sent_len, k_sent_len)

        # (num_head*batch_size, q_sent_len, k_sent_len). after softmax, the place of -np.inf will be converted to 0
        soft_attention = F.softmax(attention_matrix, dim=-1)
        soft_attention = self.dropout(soft_attention)

        output = torch.bmm(soft_attention, v)  # (batch_size, q_sent_len, v_dim)
        return output, soft_attention


@create_attention_name('scaled_dot')
class ScaledDotProductAttention(BaseAttention):
    def __init__(self, dropout_rate=0.0, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, attention_mask):
        scale = torch.tensor(k.size(-1), dtype=torch.float32)
        # for each ele in q_sent, we will compute the sim with all eles of k_sent. and we can get the sim-weight
        attention_matrix = q.bmm(k.permute(0, 2, 1).contiguous())

        attention_matrix = attention_matrix / torch.sqrt(scale)
        # or we can get attention_matrix by the following:
        # attention_matrix = attention_matrix*k.size(-1)**-0.5
        # attention_matrix = attention_matrix/k.size(-1)**0.5

        if attention_mask is not None:
            attention_matrix.masked_fill_(attention_mask, -np.inf)

        # softmax the sim-weight
        soft_attention = F.softmax(attention_matrix, dim=-1)
        soft_attention = self.dropout(soft_attention)

        output = soft_attention.bmm(v)
        return output, soft_attention


@create_attention_name('cosine')
class CosineAttention(BaseAttention):
    def __init__(self, dropout_rate=0.0, eps=1e-10, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.eps = eps

    def forward(self, q, k, v, attention_mask):
        # norm实际上在对数据进行2-范数计算。该语句是为了实现对原始特征向量的规范化
        q = q / (torch.norm(input=q, p=2, dim=-1, keepdim=True) + self.eps)
        k = k / (torch.norm(input=k, p=2, dim=-1, keepdim=True) + self.eps)

        # matmul(shape(2,3,4), shape(2,4,5))=shape(2,3,5)
        attention_matrix = torch.matmul(q, k.permute(0, 2, 1).contiguous())

        if attention_mask is not None:
            attention_matrix.masked_fill_(attention_mask, -np.inf)

        soft_attention = F.softmax(attention_matrix, dim=-1)
        soft_attention = self.dropout(soft_attention)

        output = soft_attention.bmm(v)
        return output, soft_attention


@create_attention_name('general')
class GeneralAttention(BaseAttention):
    def __init__(self, q_dim, k_dim, dropout_rate=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(q_dim, k_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, q, k, v, attention_mask=None):
        # we also can use transpose instead of permute
        attention_matrix = q.matmul(self.weight).bmm(k.transpose(1, 2).contiguous())

        if attention_mask is not None:
            attention_matrix.masked_fill_(attention_mask, -np.inf)
            # masked_fill_() change the value self but masked_fill copy
            # attention_matrix = attention_matrix.masked_fill(attention_mask, -np.inf)

        soft_attention = F.softmax(attention_matrix, dim=-1)
        soft_attention = self.dropout(soft_attention)

        output = soft_attention.bmm(v)
        return output, soft_attention

# we will update the other attentions, please looking forward for more.

'''
@create_attention_name('co_atten')
class CoAttention(BaseAttention):
    def __init__(self):
        super().__init__()
'''