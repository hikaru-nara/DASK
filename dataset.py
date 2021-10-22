import os, numpy as np
from os.path import join
import random
# import scipy.sparse as sp
# from gensim.corpora import Dictionary as gensim_dico
from brain import KnowledgeGraph
from brain.config import *
from utils.utils import is_in
import operator
import re
# import xml.etree.ElementTree as ET
# from wrapper_tokenizer import gpt2_tokenizer
from transformers import BertTokenizer, RobertaTokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import math
import torch
from numpy.random import default_rng
import time

def padding_batch(batch_data, seq_length):
    # print(batch_data)
    keys = list(batch_data[0].keys())
    lengths = [len(datum['tokens']) for datum in batch_data]
    if 'aug_tokens' in keys:
        aug_lengths = [len(datum['aug_tokens']) for datum in batch_data]
    else:
        aug_lengths = []
    max_len = seq_length
    for i in range(len(batch_data)):
        self_len = lengths[i]        
        batch_data[i]['tokens'].extend([0 for _ in range(max_len - self_len)])
        batch_data[i]['mask'].extend([0 for _ in range(max_len - self_len)])
        if 'aug_tokens' in keys:
            self_aug_len = aug_lengths[i]
            batch_data[i]['aug_tokens'].extend([0 for _ in range(max_len - self_aug_len)])
            batch_data[i]['aug_mask'].extend([0 for _ in range(max_len - self_aug_len)])
    return batch_data


def collate_fn_eval(data_list, seq_length=256):
    # print(data_list)
    keys = list(data_list[0].keys())
    batch_data = padding_batch(data_list, seq_length)
    data = {}
    data['text'] = [datum['text'] for datum in batch_data]
    
    data['tokens'] = torch.stack([torch.tensor(datum['tokens']) for datum in batch_data], dim=0)
    data['mask'] = torch.stack([torch.tensor(datum['mask']) for datum in batch_data], dim=0)
    if 'domain' in data_list[0].keys():
        data['domain'] = torch.stack([torch.tensor(datum['domain']) for datum in batch_data], dim=0)
    if 'label' in data_list[0].keys():
        data['label'] = torch.stack([torch.tensor(datum['label']) for datum in batch_data], dim=0)
    data['pos'] = None
    data['vm'] = None
    return data


def collate_fn_SSL_eval(data_list, seq_length=256):
    # print(data_list)
    keys = list(data_list[0].keys())
    batch_data = padding_batch(data_list, seq_length)
    data = {}
    data['text'] = [datum['text'] for datum in batch_data]
    
    data['tokens'] = torch.stack([torch.tensor(datum['tokens']) for datum in batch_data], dim=0)
    data['mask'] = torch.stack([torch.tensor(datum['mask']) for datum in batch_data], dim=0)
    if 'aug_tokens' in keys:
        data['aug_text'] = [datum['aug_text'] for datum in batch_data]
        data['aug_tokens'] = torch.stack([torch.tensor(datum['aug_tokens']) for datum in batch_data], dim=0)
        data['aug_mask'] = torch.stack([torch.tensor(datum['aug_mask']) for datum in batch_data], dim=0)
    if 'domain' in keys:
        data['domain'] = torch.stack([torch.tensor(datum['domain']) for datum in batch_data], dim=0)
    if 'label' in data_list[0].keys():
        data['label'] = torch.stack([torch.tensor(datum['label']) for datum in batch_data], dim=0)
    return data


def collate_fn_SSL_dev(data_list):
    source_data_list = [datum[0] for datum in data_list]
    target_data_list = [datum[1] for datum in data_list]
    return collate_fn_SSL_eval(source_data_list), collate_fn_SSL_eval(target_data_list)

def collate_fn_SSL_train(data_list):
    labeled_data_list = [datum[0] for datum in data_list]
    unlabeled_src_data_list = [datum[1] for datum in data_list]
    unlabeled_tgt_data_list = [datum[2] for datum in data_list]
    return collate_fn_SSL_eval(labeled_data_list), collate_fn_SSL_eval(unlabeled_src_data_list), \
            collate_fn_SSL_eval(unlabeled_tgt_data_list)


def bert_preprocess(datum, max_seq_length, tokenizer):
    tokens = tokenizer.encode(datum['text'], max_length=max_seq_length, add_special_tokens=True, truncation=True)
    tokens.extend([PAD_ID for _ in range(max_seq_length-len(tokens))])
    datum['tokens'] = np.array(tokens)
    datum['mask'] = np.array([1 if t!=PAD_ID else 0 for t in datum['tokens']])
    return datum

def kbert_preprocess(datum, max_seq_length, kg, return_ssl_mask=False):
    if return_ssl_mask:
        token_list, position_list, visible_matrix, ssl_vm, abs_src_pos = kg.add_knowledge_with_vm(datum['text'], \
            max_length=max_seq_length, add_special_tokens=True, return_ssl_mask=return_ssl_mask)
    else:
        token_list, position_list, visible_matrix, _ = kg.add_knowledge_with_vm(datum['text'], \
            max_length=max_seq_length, add_special_tokens=True, return_ssl_mask=return_ssl_mask)
    # token_list = [CLS_ID] + token_list[:-2] + [SEP_ID]
    mask = np.array([1 if t != PAD_ID else 0 for t in token_list])
    datum['tokens'] = np.array(token_list)
    datum['pos'] = np.array(position_list)
    datum['vm'] = visible_matrix
    datum['mask'] = mask
    if return_ssl_mask:
        datum['vm_ssl'] = ssl_vm
        datum['src_pos'] = abs_src_pos
    return datum

def SSL_preprocess(datum, max_seq_length, memory_bank, tokenizer):
    tokens = tokenizer.encode(datum['text'], add_special_tokens=True, max_length=max_seq_length, truncation=True)
    ssl_label = [-1] * len(tokens)
    for w,t in memory_bank.pivot2token.items():
        start_pos = is_in(t, tokens)
        if start_pos != len(tokens):
            ssl_label[start_pos: start_pos+len(t)] = t

    assert len(tokens) == len(ssl_label)
    # add [MASK] tag
    for i, l in enumerate(ssl_label):
        if l > 0:
            tokens[i] = MASK_ID 
    datum['tokens_org'] = tokens
    datum['ssl_label'] = ssl_label
    return datum

def Masked_SSL_preprocess(datum, max_seq_length, memory_bank, tokenizer):
    # tokens = tokenizer.encode(datum['text'], add_special_tokens=True, max_length=max_seq_length, truncation=True)
    
    tokens = datum['tokens'].tolist()
    
    ssl_label = [-1] * len(tokens)
    for w,t in memory_bank.pivot2token.items():
        start_pos = is_in(t, tokens)
        if start_pos != len(tokens) and start_pos in datum['src_pos']:
            ssl_label[start_pos: start_pos+len(t)] = t

    assert len(tokens) == len(ssl_label)
    # add [MASK] tag
    for i, l in enumerate(ssl_label):
        if l > 0:
            tokens[i] = MASK_ID 
    datum['tokens_mask'] = np.array(tokens)
    datum['ssl_label'] = np.array(ssl_label)
    datum['src_pos'] = np.array(datum['src_pos'])
    return datum


class DA_train_dataset(torch.utils.data.Dataset):
    # incomplete
    def __init__(self, labeled_data, unlabeled_data, max_seq_length, kg, model_name='bert', debug=False):
        super(DA_train_dataset, self).__init__()
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data
        self.len_labeled = len(self.labeled_data['text'])
        self.len_unlabeled = len(self.unlabeled_data['text'])
        self.max_seq_length = max_seq_length
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif model_name == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.kg = kg
        self.debug = debug
        

    def __len__(self):
        return max(self.len_labeled, self.len_unlabeled)
        # return self.len_labeled

    def __getitem__(self, i):
        assert self.len_labeled<self.len_unlabeled
        l_ind = i*self.len_labeled//self.len_unlabeled
        # l_ind = i

        labeled_datum = {k: self.labeled_data[k][l_ind] for k in self.labeled_data.keys()}
        if self.debug:
            # labeled_datum['text'] = 'Repulsive We have to do better than this'
            print(labeled_datum)

        if self.kg is None:
            labeled_datum = bert_preprocess(labeled_datum, self.max_seq_length, self.tokenizer)
        else:
            labeled_datum = kbert_preprocess(labeled_datum, self.max_seq_length, self.kg)
        if self.debug:
            print(labeled_datum.keys())
            for k in labeled_datum:
                print(k)
                print(labeled_datum[k])
                if k == 'tokens':
                    print(self.tokenizer.decode(labeled_datum[k]))
            input()
        unlabeled_datum = {k: self.unlabeled_data[k][i] for k in self.unlabeled_data.keys()}
        if self.kg is None:
            unlabeled_datum = bert_preprocess(unlabeled_datum, self.max_seq_length, self.tokenizer)
        else:
            unlabeled_datum = kbert_preprocess(unlabeled_datum, self.max_seq_length, self.kg)
        
        return labeled_datum, unlabeled_datum


class DA_test_dataset(torch.utils.data.Dataset):
    # incomplete
    def __init__(self, labeled_data, max_seq_length, kg, model_name='bert', debug=False):
        super(DA_test_dataset, self).__init__()
        self.labeled_data = labeled_data
        self.max_seq_length = max_seq_length
        if model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif model_name == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.kg = kg
        # self.augmenter = augmenter

    def __len__(self):
        return len(self.labeled_data['text'])

    def __getitem__(self, i):
        datum = {k: self.labeled_data[k][i] for k in self.labeled_data.keys()}
        if self.kg is None:
            datum = bert_preprocess(datum, self.max_seq_length, self.tokenizer)
        else:
            datum = kbert_preprocess(datum, self.max_seq_length, self.kg)
        return datum


class DA_Dataset(torch.utils.data.Dataset):
    '''
    @ Tian Li
    '''

    def __init__(self, args, source_reader, target_reader, graph_path=None, predicate=True, memory_bank=None, use_custom_vocab=None):
        super(DA_Dataset, self).__init__()
        self.source_data = source_reader.read_data()  # return a dict {'labeled':data, 'unlabeled':data}
        self.target_data = target_reader.read_data()
        self.max_seq_length = args.seq_length
        self.memory_bank = memory_bank
        if self.memory_bank is not None:
            self.memory_bank.initialize(self.source_data, self.target_data)
        # self.augmenter = augment_factory[args.augmenter](args)
        self.vocab = None
        if args.use_kg:
            self.kg = KnowledgeGraph(args, graph_path, predicate=predicate, memory_bank=memory_bank, vocab=self.vocab)
        else:
            self.kg = None
        self.model_name = 'roberta' if 'roberta' in args.model_name else 'bert'
        self.debug = args.debug
        print('modelname dataset: ', self.model_name)

    def split(self):
        labeled_src = self.source_data['labeled']
        keys = self.source_data['unlabeled'].keys()
        unlabeled = {k: self.source_data['unlabeled'][k] + self.target_data['unlabeled'][k] for k in keys}
        labeled_tgt = self.target_data['labeled']

        len_dev = len(labeled_src['text'])

        inds = list(range(len_dev))
        random.shuffle(inds)
        labeled_src = {k:[labeled_src[k][i] for i in inds] for k in labeled_src.keys()}

        dev_data = {k:labeled_src[k][len_dev//5*4:] for k in labeled_src.keys()}
        train_labeled = {k:labeled_src[k][:len_dev//5*4] for k in labeled_src.keys()}
        return DA_train_dataset(train_labeled, unlabeled, self.max_seq_length, self.kg, self.model_name, debug=self.debug), \
                DA_test_dataset(dev_data, self.max_seq_length, self.kg, self.model_name, debug=self.debug), \
                DA_test_dataset(labeled_tgt, self.max_seq_length, self.kg, self.model_name, debug=self.debug)


class DA_SSL_train_dataset(torch.utils.data.Dataset):
    def __init__(self, source_labeled, source_unlabeled, target_unlabeled, max_seq_length, kg, memory_bank):
        super(DA_SSL_train_dataset, self).__init__()
        self.source_labeled = source_labeled
        self.source_unlabeled = source_unlabeled
        self.target_unlabeled = target_unlabeled
        self.max_seq_length = max_seq_length
        self.kg = kg
        self.memory_bank = memory_bank
        self.sl_len = len(self.source_labeled['text'])
        self.su_len = len(self.source_unlabeled['text'])
        self.tu_len = len(self.target_unlabeled['text'])
        self.length = max(self.su_len, self.tu_len)
        print('get tokenizer 334')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print('got tokenizer 336')

    def __len__(self):
        return self.length 
        # return 2

    def __getitem__(self, index):
        import time
        s = time.time()
        sl_idx = int(index/self.length * self.sl_len)
        # print(self.sl_len, self.su_len, self.tu_len, index)
        if self.su_len < self.tu_len:
            su_idx = int(index/self.length * self.su_len)
            tu_idx = index
        else:
            tu_idx = int(index/self.length * self.tu_len)
            su_idx = index
        # print(su_idx, tu_idx)
        labeled_datum = {k: self.source_labeled[k][sl_idx] for k in self.source_labeled.keys()}
        if self.kg is None:
            labeled_datum = bert_preprocess(labeled_datum, self.max_seq_length, self.tokenizer)
        else:
            labeled_datum = kbert_preprocess(labeled_datum, self.max_seq_length, self.kg)
        labeled_datum = SSL_preprocess(labeled_datum, self.max_seq_length, self.memory_bank, self.tokenizer)

        src_unlabeled_datum = {k: self.source_unlabeled[k][su_idx] for k in self.source_unlabeled.keys()}
        if self.kg is None:
            src_unlabeled_datum = bert_preprocess(src_unlabeled_datum, self.max_seq_length, self.tokenizer)
        else:
            src_unlabeled_datum = kbert_preprocess(src_unlabeled_datum, self.max_seq_length, self.kg)
        src_unlabeled_datum = SSL_preprocess(src_unlabeled_datum, self.max_seq_length, self.memory_bank, self.tokenizer)

        tgt_unlabeled_datum = {k: self.target_unlabeled[k][tu_idx] for k in self.target_unlabeled.keys()}
        if self.kg is None:
            tgt_unlabeled_datum = bert_preprocess(tgt_unlabeled_datum, self.max_seq_length, self.tokenizer)
        else:
            tgt_unlabeled_datum = kbert_preprocess(tgt_unlabeled_datum, self.max_seq_length, self.kg)
        tgt_unlabeled_datum = SSL_preprocess(tgt_unlabeled_datum, self.max_seq_length, self.memory_bank, self.tokenizer)
        
        return labeled_datum, src_unlabeled_datum, tgt_unlabeled_datum


class DA_SSL_eval_dataset(torch.utils.data.Dataset):
    def __init__(self, labeled, max_seq_length, kg, memory_bank):
        super(DA_SSL_eval_dataset, self).__init__()
        self.labeled = labeled
        self.max_seq_length = max_seq_length
        self.kg = kg
        self.memory_bank = memory_bank
        print('get tokenizer 384')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print('got tokenizer 386')

    def __len__(self):
        return len(self.labeled['text'])

    def __getitem__(self, index):
        datum = {k: self.labeled[k][index] for k in self.labeled.keys()}
        if self.kg is None:
            datum = bert_preprocess(datum, self.max_seq_length, self.tokenizer)
        else:
            datum = kbert_preprocess(datum, self.max_seq_length, self.kg)
        datum = SSL_preprocess(datum, self.max_seq_length, self.memory_bank, self.tokenizer)
        return datum


class DA_SSL_dataset(torch.utils.data.Dataset):
    def __init__(self, args, source_reader, target_reader, graph_path, memory_bank, predicate=True):
        super(DA_SSL_dataset,self).__init__()
        self.source_data = source_reader.read_data()
        self.target_data = target_reader.read_data()
        self.max_seq_length = args.seq_length
        self.memory_bank = memory_bank

        memory_bank.initialize(self.source_data, self.target_data)
        if args.use_kg:
            self.kg = KnowledgeGraph(args, graph_path, predicate=predicate, vocab=None, memory_bank=memory_bank)
        else:
            self.kg = None

    def split(self):
        labeled_src = self.source_data['labeled']
        # keys = self.source_data['unlabeled'].keys()
        # unlabeled = {k: self.source_data['unlabeled'][k] + self.target_data['unlabeled'][k] for k in keys}
        unlabeled_src = self.source_data['unlabeled']
        unlabeled_tgt = self.target_data['unlabeled']
        labeled_tgt = self.target_data['labeled']

        len_dev = len(labeled_src['text'])
        # inds = list(range(len_dev))
        # random.shuffle(inds)
        # labeled_src = {k:[labeled_src[k][i] for i in inds] for k in labeled_src.keys()}
        dev_data = {k:labeled_src[k][len_dev//5*4:] for k in labeled_src.keys()}
        train_labeled = {k:labeled_src[k][:len_dev//5*4] for k in labeled_src.keys()}
        return DA_SSL_train_dataset(train_labeled, unlabeled_src, unlabeled_tgt, self.max_seq_length, self.kg, self.memory_bank), \
                DA_SSL_eval_dataset(dev_data, self.max_seq_length, self.kg, self.memory_bank), \
                DA_SSL_eval_dataset(labeled_tgt, self.max_seq_length, self.kg, self.memory_bank)


class Masked_DA_SSL_train_dataset(torch.utils.data.Dataset):
    def __init__(self, source_labeled, source_unlabeled, target_unlabeled, max_seq_length, kg, memory_bank):
        super(Masked_DA_SSL_train_dataset, self).__init__()
        self.source_labeled = source_labeled
        self.source_unlabeled = source_unlabeled
        self.target_unlabeled = target_unlabeled
        self.max_seq_length = max_seq_length
        self.kg = kg
        self.memory_bank = memory_bank
        self.sl_len = len(self.source_labeled['text'])
        self.su_len = len(self.source_unlabeled['text'])
        self.tu_len = len(self.target_unlabeled['text'])
        self.length = max(self.su_len, self.tu_len)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return self.length 
        # return 2

    def __getitem__(self, index):
        import time
        s = time.time()
        sl_idx = int(index/self.length * self.sl_len)
        # print(self.sl_len, self.su_len, self.tu_len, index)
        if self.su_len < self.tu_len:
            su_idx = int(index/self.length * self.su_len)
            tu_idx = index
        else:
            tu_idx = int(index/self.length * self.tu_len)
            su_idx = index
        assert self.kg is not None
        labeled_datum = {k: self.source_labeled[k][sl_idx] for k in self.source_labeled.keys()}        

        labeled_datum = kbert_preprocess(labeled_datum, self.max_seq_length, self.kg, return_ssl_mask=True)
        labeled_datum = Masked_SSL_preprocess(labeled_datum, self.max_seq_length, self.memory_bank, self.tokenizer)
        src_unlabeled_datum = {k: self.source_unlabeled[k][su_idx] for k in self.source_unlabeled.keys()}        
        src_unlabeled_datum = kbert_preprocess(src_unlabeled_datum, self.max_seq_length, self.kg, return_ssl_mask=True)
        src_unlabeled_datum = Masked_SSL_preprocess(src_unlabeled_datum, self.max_seq_length, self.memory_bank, self.tokenizer)
        tgt_unlabeled_datum = {k: self.target_unlabeled[k][tu_idx] for k in self.target_unlabeled.keys()}        
        tgt_unlabeled_datum = kbert_preprocess(tgt_unlabeled_datum, self.max_seq_length, self.kg, return_ssl_mask=True)
        tgt_unlabeled_datum = Masked_SSL_preprocess(tgt_unlabeled_datum, self.max_seq_length, self.memory_bank, self.tokenizer)
        labeled_datum.pop('src_pos')
        src_unlabeled_datum.pop('src_pos')
        tgt_unlabeled_datum.pop('src_pos')
        
        return labeled_datum, src_unlabeled_datum, tgt_unlabeled_datum
        # 'tokens', 'labels', 'vm', 'pos', 'tokens_mask', 'vm_ssl', 'ssl_label', 'mask'




class Masked_DA_SSL_eval_dataset(torch.utils.data.Dataset):
    def __init__(self, labeled, max_seq_length, kg, memory_bank):
        super(Masked_DA_SSL_eval_dataset, self).__init__()
        self.labeled = labeled
        self.max_seq_length = max_seq_length
        self.kg = kg
        self.memory_bank = memory_bank
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.labeled['text'])

    def __getitem__(self, index):
        datum = {k: self.labeled[k][index] for k in self.labeled.keys()}
        if self.kg is None:
            datum = bert_preprocess(datum, self.max_seq_length, self.tokenizer)
        else:
            datum = kbert_preprocess(datum, self.max_seq_length, self.kg)
        datum = SSL_preprocess(datum, self.max_seq_length, self.memory_bank, self.tokenizer)
        return datum


class Masked_DA_SSL_dataset(torch.utils.data.Dataset):
    def __init__(self, args, source_reader, target_reader, graph_path, memory_bank, predicate=True):
        super(Masked_DA_SSL_dataset,self).__init__()
        self.source_data = source_reader.read_data()
        self.target_data = target_reader.read_data()
        self.max_seq_length = args.seq_length
        self.memory_bank = memory_bank

        memory_bank.initialize(self.source_data, self.target_data)
        if args.use_kg:
            self.kg = KnowledgeGraph(args, graph_path, predicate=predicate, vocab=None, memory_bank=memory_bank)
        else:
            self.kg = None


    def split(self):
        labeled_src = self.source_data['labeled']
        # keys = self.source_data['unlabeled'].keys()
        # unlabeled = {k: self.source_data['unlabeled'][k] + self.target_data['unlabeled'][k] for k in keys}
        unlabeled_src = self.source_data['unlabeled']
        unlabeled_tgt = self.target_data['unlabeled']
        labeled_tgt = self.target_data['labeled']

        len_dev = len(labeled_src['text'])
        # inds = list(range(len_dev))
        # random.shuffle(inds)
        # labeled_src = {k:[labeled_src[k][i] for i in inds] for k in labeled_src.keys()}
        dev_data = {k:labeled_src[k][len_dev//5*4:] for k in labeled_src.keys()}
        train_labeled = {k:labeled_src[k][:len_dev//5*4] for k in labeled_src.keys()}
        return Masked_DA_SSL_train_dataset(train_labeled, unlabeled_src, unlabeled_tgt, self.max_seq_length, self.kg, self.memory_bank), \
                Masked_DA_SSL_eval_dataset(dev_data, self.max_seq_length, self.kg, self.memory_bank), \
                Masked_DA_SSL_eval_dataset(labeled_tgt, self.max_seq_length, self.kg, self.memory_bank)


dataset_factory = {'domain_adaptation': DA_Dataset,
                   'DA_SSL': DA_SSL_dataset,
                   'masked_DA_SSL': Masked_DA_SSL_dataset}

if __name__ == '__main__':
    from utils.readers import reader_factory
    # # from utils.vocab import Vocab
    # from collate_fn import collate_factory_train, collate_factory_eval
    from memory_bank import MemoryBank
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--vocab_path", default=None, type=str)
    parser.add_argument('--seq_length', default=256, type=int)
    parser.add_argument('--augmenter', default='synonym_substitution')
    parser.add_argument('--aug_rate', default=0.7, help='aug_rate for synonym_substitution')
    parser.add_argument('--use_kg', action='store_true')
    parser.add_argument('--model_name', default='base_DA', type=str)
    parser.add_argument('--pos_require_knowledge', type=str, help='the part of speech that \
        requires kg to add knowledge, choose a subset from [ADJ, ADP, ADV, CONJ, DET, NOUN, \
        NUM, PRT, PRON, VERB, ., X], split with "," e.g. ADJ,ADP,ADV', default='ADJ,ADV,NOUN')
    parser.add_argument('--use_pivot_kg', action='store_true')
    parser.add_argument('--num_pivots', type=int, default=500)
    parser.add_argument('--min_occur', type=int, default=10)
    parser.add_argument('--update_steps', type=int, default=10)
    parser.add_argument('--dataset')
    parser.add_argument('--task')
    parser.add_argument('--source')
    parser.add_argument('--target')
    parser.add_argument('--update_rate', type=float, default=0.01)
    parser.add_argument('--confidence_threshold', type=float, default=0.9)
    parser.add_argument('--filter',default='default')
    parser.add_argument('--kg_path', default='')
    parser.add_argument('--filter_conf', type=float, default=0.1)

    # parser.add_argument('--')

    args = parser.parse_args()
    if args.use_pivot_kg:
        args.vocab_require_knowledge = load_pivots(args)
    else:
        args.vocab_require_knowledge = None
    args.kg_path = args.kg_path.split(',\n\t')
    # args.kg_path = ['data/results/books_labeled_org']

    args.pollution_rate = [0.7,0.9]
    if args.task == 'domain_adaptation' or args.task=='DA_SSL' or args.task=='masked_DA_SSL':
        source = args.source
        if '.' in source:
            # print('source')
            lst = source.split('.')
            dataset_name = lst[0]
            domain_name = lst[1]
            source_reader = reader_factory[dataset_name](domain_name, 'source')
        else:
            source_reader = reader_factory[args.source]('source')

        target = args.target
        if '.' in target:
            lst = target.split('.')
            dataset_name = lst[0]
            domain_name = lst[1]
            target_reader = reader_factory[dataset_name](domain_name, 'target')
        else:
            target_reader = reader_factory[args.target]('target')

        # if args.task == 'DA_SSL':
        memory_bank = MemoryBank(args)
        dataset = dataset_factory[args.task](args, source_reader, target_reader, graph_path=args.kg_path, memory_bank=memory_bank)

        name1 = source.split('.')[1][0] if args.source != 'airlines' else 'a'
        name2 = target.split('.')[1][0] if args.target != 'airlines' else 'a'
        filename = '{}{}_pivots.txt'.format(name1,name2)
        with open(os.path.join('data',filename), 'w') as f:
            for p in memory_bank.pivots:
                f.write(p+'\n')
        # else:
        #     dataset = dataset_factory[args.task](args, source_reader, target_reader, graph_path=args.kg_path)
        train_dataset, dev_dataset, eval_dataset = dataset.split()
        
    elif args.task == 'causal_inference' or args.task == 'sentim':
        if '.' in args.dataset:
            lst = args.dataset.split('.')
            dname, domname = lst[0], lst[1]
            data_reader = reader_factory[dname](domname, 'source')
        else:
            data_reader = reader_factory[args.dataset](args.pollution_rate, causal=(args.task=='causal_inference'))

        dataset = dataset_factory[args.task](args, data_reader, graph_path=args.kg_path)
        train_dataset, dev_dataset, eval_dataset = dataset.split()

    collate_fn_eval = collate_factory_eval[args.model_name]
    collate_fn_train = collate_factory_train[args.model_name]
    # train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # dev_sampler = torch.utils.data.RandomSampler(dev_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=0, collate_fn=collate_fn_train)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=16, num_workers=0, collate_fn=collate_fn_eval)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, num_workers=0, collate_fn=collate_fn_eval)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokens = tokenizer.encode('I love Peking University. Describe the city you live in.')
    # print(tokens)
    # nltk.download('punkt')
    from transformers import BertTokenizer
    from transformers import RobertaTokenizer
    # t = BertTokenizer.from_pretrained('bert-base-uncased')
    t = RobertaTokenizer.from_pretrained('roberta-base')
    for i, (labeled_batch,src_unlabeled_batch,_) in enumerate(train_loader):

        if i==0:
            print(labeled_batch.keys())
            print(i)
            print('labeled')
            print(labeled_batch['text'][0])
            print(t.decode(labeled_batch['tokens_mask'][0]))
            print(labeled_batch['ssl_label'][0])
            print(labeled_batch['tokens_mask'][0])
            print(labeled_batch['tokens_mask'][0][labeled_batch['ssl_label'][0]>0])
            # print(labeled_batch['vm'][0][90:107,90:107])
            # print(labeled_batch['pos'][0])
            # break
