#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/6 下午5:12
# @Author  : PeiP Liu
# @FileName: Bert_retrain.py
# @Software: PyCharm
# rf https://github.com/Louis-udm/NER-BERT-CRF/blob/master/NER_BERT_CRF.py
import os
import torch
import time
import datetime
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from BertModel import BERT_CRF_NER
from Bert_data_utils import DataProcessor, BertCRFData
from transformers import AdamW, BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
from arguments import BertArgs as s_args
from common_modules.model_evaluation import time_format
from common_modules.model_evaluation import bert_evaluate as evaluate

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

abpath = '/home/liupei/Multi_features_based_semantic_augmentation_networks_for_NER_in_TI'

def warmup_linear(x, warmup = 0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


if __name__ == "__main__":
    output_dir = s_args.output_dir
    device = s_args.device
    label2idx = s_args.label2idx
    model_list = s_args.model_list
    train_examples = s_args.train_seq_list
    train_examples_labels = s_args.train_seq_label_list
    valid_examples = s_args.valid_seq_list
    valid_examples_labels = s_args.valid_seq_label_list
    test_examples = s_args.test_seq_list
    test_examples_labels = s_args.test_seq_label_list

    # we may change the value from static configuration to the dynamic
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=s_args.model_id, type=int)
    parser.add_argument("--load_checkpoint", default=s_args.load_checkpoint, type=bool)
    parser.add_argument("--batch_size", default=s_args.batch_size, type=int)
    parser.add_argument("--max_seq_len", default=s_args.max_seq_len, type=int)
    parser.add_argument("--learning_rate", default=s_args.learning_rate, type=float)
    parser.add_argument("--weight_decay_finetune", default=s_args.weight_decay_finetune, type=float)
    parser.add_argument("--lr_crf_fc", default=s_args.lr_crf_fc, type=float)
    parser.add_argument("--weight_decay_crf_fc", default=s_args.weight_decay_crf_fc, type=float)
    parser.add_argument("--total_train_epoch", default=s_args.total_train_epoch, type=int)
    parser.add_argument("--warmup_proportion", default=s_args.warmup_proportion, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=s_args.gradient_accumulation_steps, type=int)

    d_args = parser.parse_args()
    model_id = d_args.model_id
    load_checkpoint = d_args.load_checkpoint
    batch_size = d_args.batch_size
    max_seq_len = d_args.max_seq_len
    learning_rate = d_args.learning_rate
    weight_decay_finetune = d_args.weight_decay_finetune
    lr_crf_fc = d_args.lr_crf_fc
    weight_decay_crf_fc = d_args.weight_decay_crf_fc
    total_train_epoch = d_args.total_train_epoch
    warmup_proportion = d_args.warmup_proportion
    gradient_accumulation_steps = d_args.gradient_accumulation_steps

    tokenizer = BertTokenizer.from_pretrained(abpath + '/bert-base-uncased', do_lower_case=False)
    config = BertConfig.from_pretrained(abpath + '/bert-base-uncased', output_hidden_states=True)
    bert_model = BertModel.from_pretrained(abpath + '/bert-base-uncased', config=config)
    # config = BertConfig.from_pretrained(model_list[model_id], output_hidden_states=True)
    # the missing information will be filled by other file and function
    model = BERT_CRF_NER(bert_model, label2idx, batch_size=batch_size, max_seq_len=max_seq_len, device=device)

    # you should choose to start the training from scratch or from the previous
    if load_checkpoint and os.path.exists(output_dir + 'MB_bert_crf_ner.checkpoint.pt'):
        # load the model infor from previous trained model
        checkpoint = torch.load(output_dir + 'MB_bert_crf_ner.checkpoint.pt', map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        valid_acc_pre = checkpoint['valid_acc']
        valid_f1_pre = checkpoint['valid_f1']
        pretrained_model_state = checkpoint['model_state']
        # get the current model infor(mainly model parameters). if you never train, the infor(value) is empty
        cur_model_state = model.state_dict()
        # current model gets the required information from previous trained model based on common keys
        selected_pretrained_model_state = {k: v for k, v in pretrained_model_state.items() if k in cur_model_state}
        # update the current model infor. we also can regard it as value assignment
        cur_model_state.update(selected_pretrained_model_state)
        # import the information into the model
        model.load_state_dict(cur_model_state)
        print("Load the pretrained model, epoch:", checkpoint['epoch'], 'valid_acc:', checkpoint['valid_acc'],
              'valid_f1:', checkpoint['valid_f1'])
    else:
        start_epoch = 0
        valid_acc_pre = 0
        valid_f1_pre = 0
        if not os.path.exists(output_dir + 'MB_bert_crf_ner.checkpoint.pt'):
            os.mknod(output_dir + 'MB_bert_crf_ner.checkpoint.pt')


    model.to(device=device)
    # we get all of the parameters of model
    params_list = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay = ['bias', 'LayerNorm.weight']

    """
    new_param = ['transitions', 'hid2label.weight', 'hid2label.bias']
    # if we want finetuning the params in bert,we should get the parameters by:
    # bert_params = list(model.bert_model.name_parameters)
    # and the same as crf.params and hid2label.params
    optimizer_grouped_parameters = [
        # any(): any one True return True, all False return False. not any(): all False is True, One True return False.

        # if any one ele of no_decay and new_param is not in params_list, we carry out the weight_decay
        # in other word, we only carry out the weight decay for model params excluding no_decay and new_param
        {'params': [p for n, p in params_list if not any(nd in n for nd in no_decay)
                    and not any(np in n for np in new_param)], 'weight_decay': weight_decay_finetune},
        # any one of no_decay but no new_param can be carried out
        {'params': [p for n,p in params_list if any(nd in n for nd in no_decay)
                    and not any(np in n for np in new_param )], 'weight_decay': 0.0},
        # two params in new_param can be carried out
        {'params': [p for n,p in params_list if n in ('transitions', 'hid2label.weight')],
            'lr':lr_crf_fc, 'weight_decay': weight_decay_crf_fc},
        # the rest one in new_param can be carried out
        {'params': [p for n,p in params_list if n == 'hid2label.bias'],
            'lr': lr_crf_fc, 'weight_decay': 0.0}]
    """
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params_list if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_finetune},
        {'params': [p for n, p in params_list if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # the total_train_steps referring to the loss computing
    total_train_steps = int(len(train_examples)/batch_size/gradient_accumulation_steps)*total_train_epoch
    warmup_steps = int(warmup_proportion*total_train_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

    # i_th step in all steps we have planed(from 0). And here, the ideal batch_size we want is batch_size*grad_acc_steps
    global_step_th = int(len(train_examples)/batch_size/gradient_accumulation_steps * start_epoch)

    train_dp = DataProcessor(train_examples, train_examples_labels, tokenizer, max_seq_len, label2idx)
    train_bert_crf_data = BertCRFData(train_dp.get_features())
    valid_dp = DataProcessor(valid_examples, valid_examples_labels, tokenizer, max_seq_len, label2idx)
    valid_bert_crf_data = BertCRFData(valid_dp.get_features())

    train_average_loss = []
    valid_acc_score = []
    valid_f1_score = []
    for epoch in trange(start_epoch, total_train_epoch, desc='Epoch'):
        train_loss = 0
        train_start = time.time()
        model.train()
        # clear the gradient
        model.zero_grad()
        # optimizer.zero_grad()

        # shuffle=True means that RandomSampler,we can also get the train_dataloader with random by the following:
        # train_sampler = RandomSampler(train_bert_crf_data)
        # train_dataloader = DataLoader(dataset=train_bert_crf_data, sampler=train_sampler, batch_size=batch_size
        # , collate_fn=BertCRFData.seq_tensor)
        train_dataloader = DataLoader(dataset=train_bert_crf_data, batch_size=batch_size, shuffle=True, collate_fn=BertCRFData.seq_tensor)
        batch_start = time.time()
        for step, batch in enumerate(train_dataloader):
            # we show the time cost ten by ten batches
            if step % 10 == 0 and step != 0:
                print('Ten batches cost time : {}'.format(time_format(time.time()-batch_start)))
                batch_start = time.time()

            # input and output
            batch_data = tuple(cat_data.to(device) for cat_data in batch)
            train_input_ids, train_atten_mask, train_seg_ids, first_label_mask,true_labels_ids,true_label_mask=batch_data
            object_loss = model.neg_log_likehood(train_input_ids, train_atten_mask, train_seg_ids, first_label_mask,
                                                 true_labels_ids,true_label_mask)
            # train_input_ids, train_atten_mask, train_seg_ids,
            # word_token_num, true_labels_ids, true_label_mask = batch_data
            # object_loss = model.neg_log_likehood(train_input_ids, train_atten_mask, train_seg_ids, word_token_num,
            #                                     true_labels_ids, true_label_mask)

            # loss regularization
            if gradient_accumulation_steps > 1:
                object_loss = object_loss / gradient_accumulation_steps

            # Implementation of backpropagation
            object_loss.backward()
            train_loss = train_loss + object_loss.cpu().item()
            if (step+1) % gradient_accumulation_steps == 0:
                # clip the norm of gradient
                # if hasattr(optimizer, 'clip_grad_norm'):
                #     optimizer.clip_grad_norm(max_norm=1.0)
                # elif hasattr(model, 'clip_grad_norm'):
                #     model.clip_grad_norm(max_norm=1.0)
                # else:
                #     torch.nn.utils.clip_grad_norm(optimizer_grouped_parameters, max_norm=1.0)
                # this is to help prevent the "exploding gradient" problem. We have the L2 paradigm
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0, norm_type=2)
                # update the params
                optimizer.step()
                # modifying and update the learning rate with warm up which bert uses
                # lr_this_step = learning_rate*warmup_linear(global_step_th/total_train_steps)
                # # for the params in optimizer, we update the learning rate
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr_this_step
                # updating learning rate schedule
                lr_scheduler.step()
                # clear the gradient
                model.zero_grad()
                # optimizer.zero_grad()
                # # crease the i_th step
                global_step_th = global_step_th + 1
            print("Epoch:{}-{}/{}, Object-loss:{}".format(epoch, step, len(train_dataloader), object_loss))
        ave_loss = train_loss / len(train_dataloader)
        train_average_loss.append(ave_loss)

        print("Epoch: {} is completed, the average loss is: {}, spend: {}".format(epoch, ave_loss, time_format(time.time()-train_start)))
        print("***********************Let us begin the validation of epoch {}******************************".format(epoch))

        valid_dataloader = DataLoader(dataset=valid_bert_crf_data, batch_size=batch_size, shuffle=True, collate_fn=BertCRFData.seq_tensor)
        valid_acc, valid_f1 = evaluate(model, valid_dataloader, epoch, device, 'Valid')
        valid_acc_score.append(valid_acc)
        valid_f1_score.append(valid_f1)
        # if the model can achieve SOTA performance, we will save it
        if valid_f1 > valid_f1_pre:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
                        'valid_f1': valid_f1, 'max_seq_len': max_seq_len},
                       os.path.join(output_dir + 'MB_bert_crf_ner.checkpoint.pt'))
            valid_f1_pre = valid_f1

    print("**********************************************\n"
          "********     The training is over.    ********\n"
          "**********************************************")

    # then, we will show the training and validation processing by figure.
    # set the plot style from seaborn
    sns.set(style='darkgrid')
    # increase the plot size(line width) and figure size
    sns.set(font_scale=1.5)
    plt.rcParams['figure.figsize'] = [12, 6]

    x_label = np.arange(0,total_train_epoch)
    # plot the learning curve. the params are :values, color, line-title
    line1, = plt.plot(x_label, train_average_loss, color='b', label='train_average_loss')
    line2, = plt.plot(x_label, valid_acc_score, color='r', label='valid_acc_score')
    line3, = plt.plot(x_label, valid_f1_score, color='g', label='valid_f1_score')

    # now we label the plot
    plt.title('Learning curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Acc/F1')
    # legend() shows the line-title
    plt.legend(handles=[line1, line2, line3], labels=['train_average_loss','valid_acc_score', 'valid_f1_score'], loc='best')
    plt.savefig('finetuing_MB_BERT_CRF.jpg')
    plt.show()

    # Next, we will test the model on stranger dataset
    # load the pretrained model
    checkpoint = torch.load(output_dir + 'MB_bert_crf_ner.checkpoint.pt', map_location='cpu')
    # parser the model params
    epoch = checkpoint['epoch']
    valid_f1_prev = checkpoint['valid_f1']
    valid_acc_prev = checkpoint['valid_acc']
    pretrained_model_dict = checkpoint['model_state']
    # get the model param names
    model_state_dict = model.state_dict()
    # get the params interacting between model_state_dict and pretrained_model_dict
    selected_model_state = {k: v for k, v in pretrained_model_dict.items() if k in model_state_dict}
    model_state_dict.update(selected_model_state)
    # load the params into model
    model.load_state_dict(model_state_dict)
    # show the details about loaded model
    print('Loaded the pretrained NER_BERT_CRF model, epoch:', checkpoint['epoch'],
          'valid_acc:', checkpoint['valid_acc'], 'valid_f1:',checkpoint['valid_f1'])
    model.to(device)
    test_dp = DataProcessor(test_examples, test_examples_labels, tokenizer, max_seq_len, label2idx)
    test_bert_crf_data = BertCRFData(test_dp.get_features())
    test_dataloader = DataLoader(dataset=test_bert_crf_data, batch_size=batch_size, shuffle=True,
                                  collate_fn=BertCRFData.seq_tensor)
    test_acc, test_f1 = evaluate(model, test_dataloader, epoch, device, 'Test')

    """
    model.eval()
    with torch.no_grad():
        demon_dataloader = DataLoader(dataset=test_bert_crf_data, batch_size=10, shuffle=False, num_workers=4, 
                                      collate_fn=BertCRFData.seq_tensor)
        for demon_batch in demon_dataloader:
            demon_batch_data = tuple(t.to(device) for t in demon_batch)
            demon_input_ids, demon_atten_mask, demon_seg_ids, demon_pre_mask, demon_label_ids, demon_label_mask = \
                demon_batch_data
            _, demon_predicted_labels_seq_ids = model(demon_input_ids, demon_atten_mask, demon_seg_ids, demon_pre_mask, demon_label_mask)
            demon_valid_pred = torch.masked_select(demon_predicted_labels_seq_ids, demon_label_mask)
            # show the former 10 predicted examples
            for i in range(10):
                # predicted i_th example result
                print(demon_predicted_labels_seq_ids[i])
                # i_th ground_truth result
                print(demon_label_ids[i])
                # only show the ele of predicted result which the pre_mask value is true(or 1)
                valid_position_tag = demon_predicted_labels_seq_ids[i].cpu().numpy()[demon_label_mask[i].cpu().numpy() == 1]
                # show the predicted tag, not the tag ids
                print(list(map(lambda i: idx2label[i], valid_position_tag)))
                # show the true tag
                print(test_examples_labels[i])
    """
