#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/6 上午10:31
# @Author  : PeiP Liu
# @FileName: char_CNN.py
# @Software: PyCharm
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_char_dim, word_maxlen):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_char_dim,  # input_char_dim, 30
                out_channels=20,  # output_char_dim
                kernel_size=(1, 3)  # (height_filter_size, width_filter_size)=(sentlen_filter_size, wordlen_filter_size)
            ),  # the result size is (batch_size, output_char_dim, sent_len, word_len-3+1)
            nn.BatchNorm2d(20),
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=30,  # the output_char_dim = 30
                kernel_size=(1, 2)
            ),  # the result size is (batch_size, 30, sent, word_len-3)
            # nn.BatchNorm2d(30),  # the parameter is the output_channels of CNN
            nn.MaxPool2d([1, word_maxlen-3]),  # word_maxlen-3 =word_maxlen-3+1-2+1
            # we also can use the following for MaxPool2d()
            # torch.squeeze(torch.max(conv_output, -1)[0])
            nn.Dropout(0.5),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )

        self.mlp = nn.Linear(30, input_char_dim)
        self.cnn_dropout = nn.Dropout(0.5)
        self.cnn_relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # the result size of x is (batch_size, 30, sent_len, 1)
        x = x.squeeze(-1).permute(0, 2, 1)
        x = self.mlp(x)
        x = self.cnn_dropout(x)
        x = self.cnn_relu(x)
        return x
