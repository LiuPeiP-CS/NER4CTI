#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/1 上午09:23
# @Author  : PeiP Liu
# @FileName: BertModel.py
# @Software: PyCharm
# rf https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/

import os
import numpy as np
import torch
import time
import datetime
import random
import transformers
from tqdm import tqdm,trange
from transformers import BertForTokenClassification, BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from .tokenize_sentence import tokenize_sentence, tokenized_dataset, padding_sequence, split_dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

import seaborn as sns
import matplotlib.pyplot as plt


# assign the gpu for this model
os.environ['CUDA_VISIBLE_DEVICE'] = '0'

# we also can use the following to get info about cuda
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_gpu = torch.cuda.device_count()
# gpu_names = torch.cuda.get_device_name()

label2index = {'B-PER':1}
index2label = {1:'B-PER'}

# if you want reproduce the code, you can get the same random from below
# seed_num = 30
# random.seed(seed_num)
# np.random.seed(seed_num)
# torch.manual_seed(seed_num)
# torch.cuda.manual_seed_all(seed_num)


def finetuning_optimizer(full_finetuing, model):
    if full_finetuing:
        optimizer_params = list(model.named_parameters())
        no_decay = {'bias', 'gamma', 'beta'}
        # finetuning all the parameters except for no_decay
        # any(): any one True return True, all False return False. not any(): all False is True, One True return False.
        optimizer_grouped_parameters = [
            # all the eles of no_depay are not in params_list, we update the params. i.e. we do not update the no_dacay.
            # the weight_dacay_rate is a regular-item used for reducing the effect of unimportant params
            {"params": [p for n, p in optimizer_params if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in optimizer_params if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    else:
        # only finetuning the classifier parameters
        optimizer_params = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {"params": [p for n, p in optimizer_params]}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8) # eps is used for preventing denominator being 0
    return optimizer


def decay_lr(num_epoch, optimizer):
    train_dataloader, _ = split_dataset() # there are several params here...
    total_steps = len(train_dataloader) * num_epoch
    # scheduler is for learning_rate warmup. if the loss cannot convergence, the num_warmup_steps should be changed to 0
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= 0.1* total_steps, num_training_steps = total_steps)
    return lr_scheduler


def time_format(time_diff):
    seconds = int(round(time_diff))
    return str(datetime.timedelta(seconds = seconds))


if __name__ == "__main__":
    # load and construct the model for NER
    pretrained_model_name = 'bert-base-cased'  # this is the name of our choice for NER model
    model = BertForTokenClassification.from_pretrained(pretrained_model_name,
                                                       num_labels=len(label2index), output_attentions=False,
                                                       output_hidden_states=False)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, do_lower_case=False)
    model.cuda()
    optimizer = finetuning_optimizer(full_finetuing='True', model=model)
    lr_scheduler = decay_lr(num_epoch = 4, optimizer = optimizer)
    num_epoch = 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # if you want the model trained on several GPU, maybe the following can be got:
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model, device_ids = [0,1,2])
    # store the loss of each epoch
    loss_values = []
    validloss_values = []
    total_start = time.time()
    epoch_start_time = time.time()
    for __ in trange(num_epoch, desc = 'Epoch'):
        # the following is train info
        model.train()
        total_loss = 0
        train_dataloader, valid_dataloader = split_dataset() # there are several params here...
        batch_start_time = time.time()
        for batch_iter, batch_dataloader in enumerate(train_dataloader):
            # batch_dataloader = (batch_data_inputids, batch_label, batch_atten)
            if batch_iter % 10 == 0 and batch_iter != 0:
                print('Ten batches cost time : {}'.format(time_format(time.time()-batch_start_time)))
                batch_start_time = time.time()
            # add batch_data to gpu
            batch_data = tuple(each_cat.to(device) for each_cat in batch_dataloader)
            batch_inputids, batch_labels, batch_att_masks = batch_data
            # always clear any previous gradients before performing a backward pass
            model.zero_grad()
            # forward pass, this will return the loss because of given labels, classification scores (before SoftMax),
            # hidden_states (shape = (batch_size, sequence_length, hidden_size)) when output_hidden_states=True,
            # attentions (shape = (batch_size, num_heads, sequence_length, sequence_length)) when attentions=True.
            # Please refer to : https://huggingface.co/transformers/model_doc/bert.html
            outputs = model(batch_inputids, token_type_ids= None, attention_mask = batch_att_masks, labels = batch_labels)
            # now, we can get the loss
            loss = outputs.loss
            loss.backward()
            total_loss = total_loss + loss.cpu().item() # item() return the value of tensor
            # clip the norm of gradient
            # this is to help prevent the "exploding gradient" problem
            torch.nn.utils.clip_grad_norm(parameters = model.parameters(), max_norm = 2)
            # update the parameters
            optimizer.step()
            # update the learning rate
            lr_scheduler.step()
        ave_loss = total_loss / len(train_dataloader)
        print('The average trained loss of {} batches is {}'.format(len(train_dataloader), ave_loss))
        loss_values.append(ave_loss)
        # make the model into evaluation mode

        # the following is validation info
        model.eval()
        valid_loss, valid_acc = 0, 0
        # num_val_steps, num_val_examples = 0, 0
        predictions, true_labels = [], []
        for vbatch_iter, vbatch_dataloader in enumerate(valid_dataloader):
            vbatch_data = tuple(vbatch.to(device) for vbatch in vbatch_dataloader)
            vinputids, vlabels, vatten = vbatch_data
            # tell the model not compute or store gradients in order to save memory and speed up the validation
            with torch.no_grad():
                # forward pass, calculate logit prediction. When we only want the prediction, labels are optional
                voutputs = model(vinputids,token_type_ids = None, attention_mask = vatten, labels = vlabels)
            # we move the result to cpu
            logits = voutputs.logits # we also can use the index voutputs[1]
            scores = logits.detach().cpu().numpy() # detach() prevent the BP intent, and then copy the data to cpu; shape=(batch_size, seq_len, num_labels)
            predictions.extend(sent_pred for sent_pred in np.argmax(scores, axis=-1))  # the list of each sent pred
            groudtruth_label = vlabels.to('cpu').numpy()
            true_labels.extend(groudtruth_label)
            # compute the acc of this valid batch_data
            valid_loss = valid_loss + voutputs.loss.cpu().item()
        vave_loss = valid_loss / len(valid_dataloader)
        print('The average valid loss of {} batches is {}'.format(len(valid_dataloader), vave_loss))
        validloss_values.append(vave_loss)
        pred_tags = [w_p for s_p in predictions for w_p in s_p if index2label[w_p] != 'PAD']
        true_tags = [w_t for s_t in true_labels for w_t in s_t if index2label[w_t] != 'PAD']
        assert len(pred_tags) == true_tags
        print('The Acc of valid dataset is {}'.format(accuracy_score(np.array(true_tags), np.array(pred_tags))))
        print('The F1-Score of valid dataset is {}'.format(f1_score(true_tags, pred_tags)))
        print('One epoch cost time : {}'.format(time_format(time.time() - epoch_start_time)))
        epoch_start_time = time.time()

    print("All time of training is {}".format(time_format(time.time()-total_start)))
    # set the plot style from seaborn
    sns.set(style='darkgrid')

    # increase the plot size(line width) and figure size()
    sns.set(font_scale=1.5)
    plt.rcParams['figure.figsize'] = [12, 6]

    # plot the learning curve. the params are :values, color, line-title
    plt.plot(loss_values,'b-o','train-loss')
    plt.plot(validloss_values,'r-o','valid-loss')

    # now we label the plot
    plt.title('Learning curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # legend() shows the line-title
    plt.legend()
    plt.show()

    # also, you can use the model to apply your own text
    test_sentence = "Hello, this is a test sentence. Let's begin the attempt."
    # we can get the details about encode or encode_plus from
    # https://huggingface.co/transformers/internal/tokenization_utils.html?highlight=encode_plus
    # we can also get the following two sentence by :
    # test_input_ids = tokenizer.encode(test_sentence, add_special_tokens = False, return_tensors="pt")['input_ids'].to(device)
    test_tokenized_sentence = tokenizer.encode(test_sentence, add_special_tokens = False)
    test_input_ids = torch.tensor(test_tokenized_sentence['input_ids']).to(device)
    with torch.no_grad():
        test_output = model(test_input_ids)
    # the label-indices of predicted token sequence
    test_pred_label_indices = np.argmax(test_output.logits.to('cpu').numpy(), axis = -1)[0]
    # test_pred_label_indices = torch.max(test_output.logits.to('cpu'), -1)[0]

    # recover the piece tokens and token_labels to origin words and word_labels
    # we also can get the test_tokens by using tokenizer.tokenize
    # test_tokens = tokenizer.tokenize(test_sentence)
    test_tokens = tokenizer.convert_ids_to_tokens(test_input_ids.cpu().numpy())
    test_new_words, test_new_pred_labels = [], []
    for each_test_token, each_pred_indices in zip(test_tokens, test_pred_label_indices):
        if each_test_token.startwith("##"):
            test_new_words[-1] = test_new_words[-1] + each_test_token.replace('##','')
        else:
            test_new_words.append(each_test_token)
            test_new_pred_labels.append(index2label[each_pred_indices])

    # print the predicted result
    for word, label in zip(test_new_words, test_new_pred_labels):
        print('{}\t{}'.format(word, label))