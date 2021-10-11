#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/6 下午3:47
# @Author  : PeiP Liu
# @FileName: Transformer_module.py
# @Software: PyCharm

import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from common_modules.Attention import build_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=256, num_head=8, dropout_rate=0.0, attention_type='scaled_dot'):
        # input_shape = (batch_size, sent_len, model_dim)
        super().__init__()
        assert model_dim % num_head == 0, "model_dim should be divided by num_head"
        self.h_size = model_dim
        self.num_head = num_head
        self.each_head_size = model_dim // num_head

        self.linear_q = nn.Linear(self.h_size, self.h_size)
        self.linear_k = nn.Linear(self.h_size, self.h_size)
        self.linear_v = nn.Linear(self.h_size, self.h_size)

        self.attention = build_attention(attention_type, q_dim=self.each_head_size, k_dim=self.each_head_size)
        self.fc = nn.Linear(self.h_size, self.h_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, q, k, v, attention_mask=None):
        batch_size = q.size(0)

        residual = q

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        q = q.view(batch_size * self.num_head, -1, self.each_head_size)  # split the feature to num_head splides
        k = k.view(batch_size * self.num_head, -1, self.each_head_size)  # 将数据在特征维度进行num_head的切分
        v = v.view(batch_size * self.num_head, -1, self.each_head_size)

        if attention_mask is not None:
            # 将mask信息扩充num_head个头上,(num_head*batch_size, q_sent_len, k_sent_len)
            attention_mask = attention_mask.repeat(self.num_head, 1, 1)
        # the input_shape = q_shape = k_shape = v_shape = (num_head*batch_size, q_sent_len, v_dim(self.each_head_size))
        # the output(context) shape = (num_head*batch_size, q_sent_len, v_dim),
        # the attention shape = (num_head*batch_size, q_sent_len, k_sent_len)
        context, attention = self.attention(q, k, v, attention_mask)  # attention and concatenate

        context = context.contiguous().view(batch_size, -1, self.h_size)  # reshape into the orig size

        output = self.dropout(self.fc(context))

        output = self.layer_norm(residual + output)

        return output, attention  # output_shape = (batch_size, sent_len, model_dim)


class FullFeatMultiHeadAttention(nn.Module):
    def __init__(self, model_dim=256, num_head=8, dropout_rate=0.0, attention_type='scaled_dot'):
        super().__init__()
        self.num_head = num_head
        self.h_size = model_dim
        self.atten_type = attention_type

        self.w_q = nn.Linear(self.h_size, self.h_size * self.num_head)
        self.w_k = nn.Linear(self.h_size, self.h_size * self.num_head)
        self.w_v = nn.Linear(self.h_size, self.h_size * self.num_head)

        self.attention = build_attention(attention_type,
                                         q_dim=self.h_size * self.num_head, k_dim=self.h_size * self.num_head)

        self.fc = nn.Linear(self.h_size * self.num_head, self.h_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, q, k, v, attention_mask=None):
        batch_size = q.size(0)

        residual = q

        q = self.w_q(q).view(batch_size * self.num_head, -1, self.h_size)  # split the feature to num_head slides
        k = self.w_k(k).view(batch_size * self.num_head, -1, self.h_size)  # 将数据在特征维度进行num_head的切分
        v = self.w_v(v).view(batch_size * self.num_head, -1, self.h_size)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(self.num_head, 1, 1)
        context, attention = self.attention(q, k, v, attention_mask)  # attention and concatenate

        context = context.contiguous().view(batch_size, -1, self.h_size * self.num_head)  # reshape into the orig size

        output = self.dropout(self.fc(context))

        output = self.layer_norm(residual + output)  # (batch_size, seq_len, h_size)

        return output, attention


class FeedForward(nn.Module):
    def __init__(self, model_dim=256, hidden_dim=1024, dropout_rate=0.0):
        # input_shape = (batch_size, sent_len, model_dim)
        super(FeedForward, self).__init__()
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim

        self.Linear1 = nn.Linear(self.model_dim, self.hidden_dim)
        self.Linear2 = nn.Linear(self.hidden_dim, self.model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        output = self.Linear2(F.relu(self.Linear1(x)))

        output = self.dropout(output)

        output = self.norm(x + output)

        return output  # output_shape = (batch_size, sent_len, model_dim)
