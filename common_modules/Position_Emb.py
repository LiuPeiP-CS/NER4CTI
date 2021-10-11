#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/8 上午9:44
# @Author  : PeiP Liu
# @FileName: Position_Emb.py
# @Software: PyCharm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, sent_len=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('position_embedding', self.get_sinusoid_encoding_table(sent_len, model_dim))  # 相当于属性赋值

    def get_sinusoid_encoding_table(self, sent_len, model_dim):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2*(model_dim_j//2)/model_dim) for model_dim_j in range(model_dim)]

        sinusoid_table = np.array([get_position_angle_vec(i_position) for i_position in range(sent_len)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim = 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim = 2i+1
        return torch.tensor(sinusoid_table, dtype=torch.float32).unsqueeze(0)  # shape = (1, sent_len, model_dim)

    def forward(self, x):
        return x + self.position_embedding[:, :x.size(1)].clone().detach()


class PositionFeedForward(nn.Module):
    def __init__(self, model_dim, model_dim_hid, dropout_rate=0.1):
        super(PositionFeedForward, self).__init__()
        self.w_in2hid = nn.Linear(model_dim, model_dim_hid)
        self.w_hid2out = nn.Linear(model_dim_hid, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-9)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x

        x = self.w_hid2out(F.relu(self.w_in2hid(x)))
        x = self.dropout(x)
        x = residual + x
        return x
