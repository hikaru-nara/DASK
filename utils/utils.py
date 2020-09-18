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



def accuracy(batch_logits, batch_labels):
    batch_size = batch_labels.shape[0]
    # pred = torch.eq(batch_logits, torch.max(batch_logits, -1, keepdim=-1))[:,0]
    pred = torch.argmax(batch_logits, -1)
    correct = torch.sum(torch.eq(pred, batch_labels))
    acc = correct/batch_size
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
        

def create_logger(cfg, log_dir, phase='train'):
    root_output_dir = log_dir
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

def get_optimizer(cfg, params):
    optimizer = None
    # specify learning rate for different groups here
    # reference https://pytorch.org/docs/stable/optim.html
    # param_dict = [{'params':param} for param in param_groups]
    if cfg.optimizer == 'sgd':
        optimizer = optim.SGD(
            [{'params':params}],
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == 'adam':
        optimizer = optim.Adam(
            [{'params':params}],
            lr=cfg.learning_rate
        )

    return optimizer