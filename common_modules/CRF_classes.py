#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/3 下午15:36
# @Author  : PeiP Liu
# @FileName: BertModel.py
# @Software: PyCharm
import torch
import torch.nn as nn


# another choice is open package from https://pytorch-crf.readthedocs.io/en/stable/
# besides, I think the introduction here is also a good choice: https://www.cnblogs.com/weilonghu/p/11960984.html

class CRF(nn.Module):
    # here, the num_labels contain bos, pad and eos
    def __init__(self, num_labels, pad_idx, bos_idx, eos_idx, device):
        super().__init__()
        # valid_tag_nums = len(tag2index)
        # for iter, new_tag in enumerate(['PAD', 'bos', 'eos']):
        #     if new_tag not in tag2index:
        #         tag2index[new_tag] = valid_tag_nums + iter
        self.num_labels = num_labels
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.device = device
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels), requires_grad=True)
        # nn.init.uniform_(self.transitions, -0.1, 0.1)
        # some impossible transfer with exp(-10000)
        self.transitions.data[:, self.bos_idx] = -10000
        self.transitions.data[self.eos_idx, :] = -10000
        self.transitions.data[self.pad_idx, :] = -10000
        self.transitions.data[:, self.pad_idx] = -10000
        self.transitions.data[self.pad_idx, self.pad_idx] = 0
        self.transitions.data[self.pad_idx, self.eos_idx] = 0

    def forward(self, features, tag_seqs, mask=None):
        if mask is None:
            mask = torch.ones(tag_seqs.shape, dtype=torch.float, device=self.device).bool()
        # the best path's score of each sentence in batch, and also is the maximum score
        max_score = self.score_sentences(features, tag_seqs, mask) # (batch_size, )
        # all possible scores of each sentence in batch
        possible_scores = self.forward_alg(features, mask) # (batch_size, )
        # the object is to minimize the result.
        return -torch.mean(max_score - possible_scores) # (1, )

    # the shape of features is (batch, seq_length, num_tags)
    def forward_alg(self, features, mask):
        # batch_size = features.shape[0]
        seq_length = features.shape[1]
        # Initial forward_var is the tensor of likelihoods combining the
        # transitions to the initial states and the emission for the first time-step.
        # (batch_size, num_labels)=(1, num_labels)+(batch_size, num_labels)
        forward_var = self.transitions[self.bos_idx, :].unsqueeze(0) + features[:, 0]
        for i_step in range(1, seq_length):
            alpha_t = []
            # every tag in the step_time: i_step
            for cur_label in range(self.num_labels):
                # emission_score = features[:, i_step, cur_label].view(batch_size, 1, -1).expand(batch_size, 1, self.num_labels)
                emission_score = features[:, i_step, cur_label].unsqueeze(1) # (Batch_size, 1)
                # transition_score = self.transitions[:cur_label].view(1, -1)
                transition_score = self.transitions[:, cur_label].unsqueeze(0) # (1, num_tags)
                # add the previous log result
                cur_label_forward_var = forward_var + emission_score + transition_score # (Batch_size, num_tags)
                # we can get the result with the label-dim, alpha_t.append(shape = (batch_size,))
                alpha_t.append(torch.logsumexp(cur_label_forward_var, dim=-1))
            # stack the infor on num_label dim, and we can get the tensor with shape = (batch_size, num_tags)
            step_forward_var = torch.stack(alpha_t, dim=1)
            new_mask = mask[:, i_step].int().unsqueeze(1) # shape=(batch_size,) ————> shape=(batch_size, 1)
            # In valid positions (mask == True) we want to take the logsumexp over the current_tag dimension
            # of 'inner'. Otherwise (mask == False) we want to retain the previous alpha. shape = (batch_size, num_tags)
            forward_var = new_mask * step_forward_var + (1-new_mask) * forward_var
        # (batch_size, num_labels)
        final_step_forward_var = forward_var + self.transitions[:, self.eos_idx].unsqueeze(0)
        return torch.logsumexp(final_step_forward_var, dim=-1) # (batch_size,)

    def score_sentences(self, features, tag_seqs, mask):
        '''
        :param features: the output from previous nn (we also can regard it as the emission), and the shape=(batch_size, seq_len, feature_dim=num_labels)
        :param tag_seqs: the best tagging sequence, and shape = (batch_size, seq_len). each ele within it is <= num_labels
        :param mask: indices the valid position within each sequence, shape = (batch_size, seq_len)
        :return: the score of provided tag sequences
        '''
        batch_size, seq_length = features.shape[0], features.shape[1]
        batch_scores = torch.zeros(batch_size, dtype = torch.float, device = self.device)
        batch_first_labels = tag_seqs[:, 0] # shape=(batch_size,)
        # we can get the infor from -1_dim of features[:,0] based on the index-ele from batch_first_labels
        emission_score = features[:, 0].gather(-1, batch_first_labels.unsqueeze(1)).squeeze() # shape = (batch_size,)
        # the ele——self.bos_idx and each ele from batch_first_labels can construct a tuple
        trans_score = self.transitions[self.bos_idx, batch_first_labels] # shape = (batch_size,)
        batch_scores = batch_scores + emission_score + trans_score
        for i_step in range(1, seq_length):
            batch_pre_labals = tag_seqs[:, i_step-1] # (batch_size,)
            batch_cur_labels = tag_seqs[:, i_step]
            # features[:,i_step].shape = (batch_size, num_labels), shape = (batch_size,)
            emission_score = features[:, i_step].gather(-1, batch_cur_labels.unsqueeze(1)).squeeze()
            # one ele from batch_pre_labels and another from batch_cur_labels construct the tuple index,
            trans_score = self.transitions[batch_pre_labals, batch_cur_labels] # shape = (batch_size,)
            # applying masking: if the position is pad, then the contribution score should be 0
            emission_score = mask[:, i_step] * emission_score
            trans_score = mask[:, i_step] * trans_score
            # get the score from previous step and current step (including transition_score and emission score)
            batch_scores = emission_score + trans_score + batch_scores
        # get the valid index (valid_length-1) of each sequence, (batch_size,)
        valid_length = mask.sum(1) - 1
        # valid_length.unsqueeze(1).shape = (batch_size,1). this can get the last valid tag of each sentence
        batch_last_valid_label = tag_seqs.gather(1, valid_length.unsqueeze(1)).squeeze() # (batch_size,)
        # get the score from the last valid tag to eos. shape = (batch_size, )
        batch_scores = batch_scores + self.transitions[batch_last_valid_label, self.eos_idx]
        return batch_scores

    def viterbi_decode(self, features, mask):
        '''
        :param features: tha same as upper introduction
        :param mask: store the valid length infor for the sequences
        :return: best sequence tagging
        '''
        back_pointers = []
        batch_size, seq_len, num_labels = features.shape
        # the first token's scores among all labels; it represents the previous step's scores
        alphas = self.transitions[self.bos_idx, :].unsqueeze(0) + features[:, 0] # shape = (batch_size, num_labels)
        for i_step in range(1, seq_len):
            alpha_i_step = []
            back_pointer_i_step = []
            for j_label in range(num_labels):
                # get the emission-score of (i_step-time, label-dim)
                emission_score = features[:, i_step, j_label].unsqueeze(1) # (batch_size, 1)
                # all the transition score from others to j_label
                transition_score = self.transitions[:, j_label].unsqueeze(0) # (1, num_labels)
                # combine current scores with previous alphas (in log space)
                scores = alphas + emission_score + transition_score # (batch_size, num_labels)
                # recording which the previous tag can lead current j_label at i_step position
                # to have maximum score, and what is the score
                ij_max_scores, ij_max_pre_labels = torch.max(scores, dim = -1) # (batch_size, )
                # record the maximum scores of all labels at i_step, and their corresponding previous labels
                alpha_i_step.append(ij_max_scores) # [(batch_size, ),(batch_size, ), ...]
                back_pointer_i_step.append(ij_max_pre_labels)
            # we aggregate the scatter list to a tensor to express this i_step
            alpha_i_step_tensor = torch.stack(alpha_i_step, dim = 1) # (batch_size, num_labels)
            # get the valid length of each sentence to retain valuable infor
            masking = mask[:, i_step].int().unsqueeze(1) # (batch_size, 1)
            # update alphas based on valid length, and retain some infor from previous step, (batch_size, num_labels)
            alphas = masking * alpha_i_step_tensor + (1-masking) * alphas
            back_pointers.append(back_pointer_i_step) # finally, (seq_len-1, num_labels, batch_size)
        # (batch_size, num_labels)=(batch_size, num_labels)+(1, num_labels). final scores on each labels of batch sentences
        final_score = alphas + self.transitions[:, self.eos_idx].unsqueeze(0)

        """
        # if all eles of mask are True:
        # then, we start backtracking for the best path
        # firstly, get the max final label. shape = (batch_size,)
        final_max_scores, final_max_pre_label = torch.max(final_score, dim = -1)
        # get the best label of each step on the path
        step_best_label = final_max_pre_label
        # record the best label of all steps on the path
        best_path = [final_max_pre_label]
        # then, backtracking from the batch_sequences tail.
        # The final back_path_step is 1-token, which contains the pre-label belonging to 0-token
        for back_path_step in reversed(back_pointers):
            # get previous labels' positions from the tuple axis=(back_path_step, step_best_label)
            step_best_pre_label = back_path_step.gather(-1, step_best_label.unsqueeze(1)).squeeze(1)
            # insert the pre_labels at the front.finally, shape = (seq_len, batch_size)
            best_path.insert(0, step_best_pre_label)
            step_best_label = step_best_pre_label
        # aggregate the tagging for each sentence, shape = (batch_size, seq_len)
        batch_best_path = torch.stack(best_path, dim = -1)
        return batch_best_path
        """
        final_max_scores, final_max_pre_label = torch.max(final_score, dim = -1)
        # decode the best sequence for current batch
        # follow the backpointers to find the best tagging path
        best_paths = []
        valid_length = mask.sum(1)

        # we get the best tagging path for each sentence one by one
        for i_seq in range(batch_size):
            # the length value of i_seq, one scalar value
            i_valid_length = valid_length[i_seq].item()
            # the final tag of i_seq, and also is the index of pre_tag(i-1). one scalar value
            i_final_max_pre_label = final_max_pre_label[i_seq].item()
            # get all the tagging of batch sentence with the end of i_length. (i_valid_length, num_label, batch_size)
            batch_tagging_i = back_pointers[:i_valid_length-1]
            i_seq_best_path = [i_final_max_pre_label]
            pre_label_index = i_final_max_pre_label
            for batch_tagging_j_step in reversed(batch_tagging_i):
                # batch_tagging_j_step.shape = (num_labels, batch_size)
                cur_best_label = batch_tagging_j_step[pre_label_index][i_seq].item()
                i_seq_best_path.insert(0, cur_best_label)
                pre_label_index = cur_best_label
            best_paths.append(i_seq_best_path)

        return final_max_scores, best_paths
