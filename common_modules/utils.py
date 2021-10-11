#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/10 下午5:02
# @Author  : PeiP Liu
# @FileName: utils.py
# @Software: PyCharm
import numpy as np


def decode_tag(emission, valid_mask):
    """
    :param emission: (batch_size, sent_len, num_labels)
    :param valid_mask: (batch_size, sent_len)
    :return:
    """
    valid_sentlen =valid_mask.sum(1)  # (batch_size,), the valid length of each sent in the batch-data
    pre_tag = emission.argmax(-1)  # (batch_size, sent_len, ), get the place of max ele in feature-dim
    pre_valid_tag = [pre_tag[i_sent][:valid_sentlen[i_sent].item()].detach().tolist() for i_sent in range(emission.size(0))]

    return pre_valid_tag


class EarlyStopping:
    def __init__(self, monitor='loss', min_delta=0, patience=0):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

        self._wait = 0
        self._best = None
        self._best_epoch = None

        if 'loss' is self.monitor:
            self.monitor_op = np.less
            self.min_delta = self.min_delta * -1
        else:
            self.monitor_op = np.greater
            self.min_delta = self.min_delta * 1

    def judge(self, epoch, value):
        current = value
        if self._best is None:
            self._best = current
            self._best_epoch = epoch
            return  # the first return empty
        if self.monitor_op(current-self.min_delta, self._best):
            self._best = current
            self._best_epoch = epoch
            self._wait = 0
            return
        else:
            self._wait = self._wait + 1
            if self._wait >= self.patience:  # when the f1_score decay in patience consecutive epoch, we stop the train
                return True

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def best_val(self):
        return self._best
