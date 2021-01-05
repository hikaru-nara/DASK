# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import six
import numpy as np 
import math
from pathlib import Path
import unidecode
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize

def extract_word_freq(sentences):
    freq_dict = {}
    for s in sentences:
        words = word_tokenize(s)
        for w in words:
            if not w in freq_dict:
                freq_dict[w] = 1
            else:
                freq_dict[w] += 1
    return freq_dict

def sentiment_score_init(source_labeled_text,source_label):
    word_count = {}
    word_sentiment_count = {}
    for sentence, label in zip(source_labeled_text,source_label):
        for word in word_tokenize(sentence):
            if word in word_count:
                word_count[word] += 1
                if label==1:
                    word_sentiment_count[word] += 1
                else:
                    word_sentiment_count[word] -= 1
            else:
                word_count[word] = 1
                word_sentiment_count[word] = label*2 - 1
    return word_count, word_sentiment_count

def is_in(pattern, seq):
    lenp = len(pattern)
    for i in range(len(seq)):
        if seq[i:i+lenp] == pattern:
            return i
    return len(seq)

def load_pivots(args):
    source = args.source
    target = args.target
    pivot_file = os.path.join('data', '{0}-{1}'.format(source, target), 'all_pivot_stats.csv')
    table = pd.read_csv(pivot_file)
    col = table.columns
    # data = {'text':[str(t) for t in table['text']]}
    pivot_num = args.num_pivots
    pivots = {'word':[],'label':[]}
    curr_num = 0
    for i in range(len(table['word'])):
        if table['src_freq'][i]>args.min_occur and table['tgt_freq'][i]>args.min_occur:
            curr_num += 1
            pivots['word'].append(table['word'][i])
            pivots['label'].append(table['polar'][i])
        if curr_num == pivot_num:
            return pivots
    return pivots


def has_digit(wordlist):
    for i in range(10):
        for j,m in enumerate(wordlist):
            if i==m:
                return j
    return False


def load_pretrain_for_two_stage_kbert(model, pretrained_state_dict):
    keys = list(pretrained_state_dict.keys())
    for k in keys:
        spk = k.split('.')
        if spk[-2]=='LayerNorm':
            spk[-1] = 'weight' if spk[-1]=='gamma' else 'bias'
        if spk[1]=='embeddings':
            spk.insert(1,'0')
        
            params = pretrained_state_dict.pop(k)
            new_K = '.'.join(spk)
            pretrained_state_dict[new_K] = params

            spk.pop(1)
            spk.insert(1,'1')
            new_K = '.'.join(spk)
            pretrained_state_dict[new_K] = params
        elif spk[1]=='encoder':
            if int(spk[3])<6:
                spk.insert(1,'0')
            else:
                spk[3] = str(int(spk[3])-6)
                spk.insert(1,'1')
            params = pretrained_state_dict.pop(k)
            new_K = '.'.join(spk)
            pretrained_state_dict[new_K] = params
    model.load_state_dict(pretrained_state_dict, strict=False)


def save_attention_mask(attentions, text, pos, tokens, log_dir):
    batch_size = len(text)
    for i in range(batch_size):
        attentions_ = [a[i,:,:,:] for a in attentions]
        text_ = text[i]
        pos_ = pos[i,...]
        tokens_ = tokens[i,...]
        with open(os.path.join(log_dir,'attentions{}.pkl'.format(i)), 'wb') as f:
            pickle.dump(attentions_, f)
        with open(os.path.join(log_dir, 'pos{}.pkl'.format(i)), 'wb') as f:
            pickle.dump(pos_, f)
        with open(os.path.join(log_dir, 'tokens{}.pkl'.format(i)), 'wb') as f:
            pickle.dump(tokens_, f)
        with open(os.path.join(log_dir, 'text{}.pkl'.format(i)), 'wb') as f:
            pickle.dump(text_, f)
        print('==>attentions and so forth are saved to {}'.format(log_dir))



def standardize(word):
    word = unidecode.unidecode(word)
    print('standardize')
    print(word)
    result = word.strip('\'')
    tmp=result.find('\'')
    if tmp!=-1:
        result = result[:tmp]
    return result

def pollute_data(t, y, pollution):
    """
    @ Invariant Rationalization, Chang and Zhang
    Pollute dataset. 
    Inputs:
        t -- texts (np array)
        y -- labels (np array)
        pollution -- a list of pollution rate for different envs
            if 2 envs total, e.g. [0.9, 0.7]
    """
    num_envs = len(pollution)

    pos_idx = np.where(y > 0.)[0]
    neg_idx = np.where(y == 0.)[0]

    # shaffle these indexs
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    # obtain how many pos & neg examples per env
    num_pos = len(pos_idx)
    num_neg = len(neg_idx)

    n = math.floor(num_pos / num_envs)
    num_pos_per_env = np.array(
        [n if i != num_envs - 1 else num_pos - n * i for i in range(num_envs)])
    assert (np.sum(num_pos_per_env) == num_pos)

    n = math.floor(num_neg / num_envs)
    num_neg_per_env = np.array(
        [n if i != num_envs - 1 else num_neg - n * i for i in range(num_envs)])
    assert (np.sum(num_neg_per_env) == num_neg)

    # obtain the pos_idx and neg_idx for each envs
    env_pos_idx = []
    env_neg_idx = []

    s = 0
    for i, num_pos in enumerate(num_pos_per_env):
        idx = pos_idx[s:s + int(num_pos)]
        env_pos_idx.append(set(idx))
        s += int(num_pos)

    s = 0
    for i, num_neg in enumerate(num_neg_per_env):
        idx = neg_idx[s:s + int(num_neg)]
        env_neg_idx.append(set(idx))
        s += int(num_neg)

    # create a lookup table idx --> env_id
    idx2env = {}

    for env_id, idxs in enumerate(env_pos_idx):
        for idx in idxs:
            idx2env[idx] = env_id
    assert (len(idx2env.keys()) == len(pos_idx))

    for env_id, idxs in enumerate(env_neg_idx):
        for idx in idxs:
            idx2env[idx] = env_id
    assert (len(idx2env.keys()) == len(t))

    new_t = []
    envs = []

    for idx, t_ in enumerate(t):
        env_id = idx2env[idx]
        rate = pollution[env_id]

        envs.append(env_id)

        if np.random.choice([0, 1], p=[1. - rate, rate]) == 1:
            if y[idx] == 1.:
                text = ", " + t_
                # text =  t_ + " ,"
            else:
                text = ". " + t_
                # text =  t_ + " ."
        else:
            if y[idx] == 1.:
                text = ". " + t_
                # text =  t_ + " ."
            else:
                text = ", " + t_
                # text =  t_ + " ,"
        new_t.append(text)

    return new_t, envs

def convert_to_unicode(text):
    """
    Converts text to Unicode (if it's not already)
    assuming utf-8 input.
    """
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def accuracy(batch_logits, batch_labels):
    '''
    @ Tian Li
    '''
    batch_size = batch_labels.shape[0]
    pred = np.argmax(batch_logits, -1)
    correct = np.sum((pred==batch_labels).astype(np.int))
    acc = correct.astype(np.float)/float(batch_size)
    return acc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        

def create_logger(log_dir, phase='train'):
    root_output_dir = Path(log_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(root_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def consistence(optimizers, model):
    consistency = True
    c = model.children().__next__()
    param_group_m = [cc.parameters() for cc in c.children()]
    param_group_o = [optimizer.param_groups[0]['params'] for optimizer in optimizers]
    for i, (pg1, pg2) in enumerate(zip(param_group_m, param_group_o)):
        for p1, p2 in zip(pg1, pg2):
            b = torch.eq(p1, p2).all()
            consistency = consistence and b and (p1 is p2)
            # print(b, p1 is p2)
    return consistency

    # consistency = True
    # print(isinstance(optimizer.param_groups, list))
    # print(isinstance(optimizer.param_groups[0], dict))
    # print(optimizer.param_groups[0].keys())
    # for p1, p2 in zip(optimizer.param_groups[0]['params'], model.parameters()):
    #     a = torch.eq(p1,p2).all()
    #     b = p1 is p2
    #     consistency = a and consistency
    #     consistency = b and consistency
    #     print(a, b, consistency)
    # return consistency

def get_optimizer(cfg, param, lr):
    optimizer = None
    # specify learning rate for different groups here
    # reference https://pytorch.org/docs/stable/optim.html
    # param_dict = [{'params':param} for param in param_groups]
    if cfg.optimizer == 'sgd':
        optimizer = optim.SGD(
            param,
            lr=lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == 'adam':
        optimizer = optim.Adam(
            param,
            lr=lr
        )

    return optimizer