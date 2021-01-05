import os, numpy as np
from os.path import join
import random
# import scipy.sparse as sp
# from gensim.corpora import Dictionary as gensim_dico
from brain.knowgraph import KnowledgeGraph
# from brain.config import *
from utils.utils import standardize, is_in
import operator
import re
# import xml.etree.ElementTree as ET
# from wrapper_tokenizer import gpt2_tokenizer
from transformers import BertTokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import math
import torch
from numpy.random import default_rng
from uer.utils.constants import *
import time
# from augmentor import augment_factory

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def create_linear_semantic_tree(kg, datum, vocab):
    # get semantic tree (i.e. token_list), position embedding and visible matrices
    text = datum['text']
    token_list, position_list, visible_matrix, _ = kg.add_knowledge_with_vm(text)
    # note that this vocab is not necessarily the same as is used in kg.tokenizer
    token_list = [CLS_TOKEN] + token_list[:-1]
    token_ids = np.array([vocab.get(t) for t in token_list])
    # print('dataset/createtree')
    # print(datum['text'])
    # print(token_list)
    # print(token_ids)
    # print(token_ids)
    mask = np.array([1 if t != PAD_TOKEN else 0 for t in token_list])
    position_array = np.array(position_list)
    datum['token'], datum['mask'], datum['pos'], datum['vm'] = token_ids, mask, position_array, visible_matrix
    return datum


# def bert_preprocess(datum, vocab, max_seq_length, tokenizer):
#     tokens = [int(t) for t in tokenizer.encode(datum['text'])]
#     # print(datum.keys())
#     new_datum = {}
#     if 'label' in datum.keys():
#         new_datum['label']=datum['label']
#     new_datum['tokens'] = ([vocab.CLS_TOKEN_ID] + tokens + [vocab.SEP_TOKEN_ID])[:max_seq_length]
#     new_datum['mask'] = [1 for _ in range(len(new_datum['tokens']))]
#     padding_len = max_seq_length - len(new_datum['tokens'])
#     mask_padding = [0 for _ in range(padding_len)]
#     token_padding = [vocab.PAD_TOKEN_ID for _ in range(padding_len)]
#     new_datum['tokens'].extend(token_padding)
#     new_datum['mask'].extend(mask_padding)
#     return {key:torch.tensor(value) for key, value in new_datum.items()}

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
    # print(datum.keys())
    # print(datum)
    # tokens = tokenizer.encode('[CLS]' + datum['text'] + '[SEP]', max_length=max_seq_length)
    tokens = tokenizer.encode(datum['text'], max_length=max_seq_length, add_special_tokens=True, truncation=True)
    tokens.extend([0 for _ in range(max_seq_length-len(tokens))])
    # if len(tokens) > max_seq_length:
    #     tokens = tokens[:max_seq_length - 1] + [tokens[-1]]

    datum['tokens'] = np.array(tokens)
    datum['mask'] = np.array([1 if t!=0 else 0 for t in datum['tokens']])
    if 'aug_text' in datum.keys():
        aug_tokens = tokenizer.encode(datum['aug_text'], max_length=max_seq_length, add_special_tokens=True, truncation=True)
        # if len(tokens) > max_seq_length:
        #     tokens = tokens[:max_seq_length - 1] + [tokens[-1]]

        datum['aug_tokens'] = aug_tokens
        datum['aug_mask'] = [1 for _ in range(len(datum['aug_tokens']))]
    # if len(tokens)<max_seq_length:
    #     padding_len = max_seq_length - len(datum['tokens'])
    #     mask_padding = [0 for _ in range(padding_len)]
    #     token_padding = [PAD_ID for _ in range(padding_len)]
    #     # print('prepricess')
    #     # print(datum['tokens'])
    #     datum['tokens'].extend(token_padding)
    #     datum['mask'].extend(mask_padding)

    # datum['tokens'], datum['mask'], datum['domain'] = torch.tensor(datum['tokens']), torch.tensor(datum['mask']), torch.tensor['domain']
    return datum

def kbert_preprocess(datum, max_seq_length, kg):
    
    token_list, position_list, visible_matrix, _ = kg.add_knowledge_with_vm(datum['text'], max_length=max_seq_length, add_special_tokens=True)
    # token_list = [CLS_ID] + token_list[:-2] + [SEP_ID]
    mask = np.array([1 if t != PAD_TOKEN else 0 for t in token_list])
    datum['tokens'] = np.array(token_list)
    datum['pos'] = np.array(position_list)
    datum['vm'] = visible_matrix
    datum['mask'] = mask
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
            tokens[i] = tokenizer.encode('[MASK]')[1] # [101, 103, 102]
    datum['tokens_org'] = tokens
    datum['ssl_label'] = ssl_label
<<<<<<< HEAD
    # datum['ssl_label'] = np.array(ssl_label)
=======
>>>>>>> dad1c79807ef321a3e31679b7a895b566d43a276
    return datum



class Causal_Train_Dataset(torch.utils.data.Dataset):
    '''
    @ Tian Li
    '''

    def __init__(self, train_data, kg, max_seq_length):
        self.train_data = train_data
        self.kg = kg
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __getitem__(self, i):
        # 0 ->i
        datum = {key: value[i] for key, value in self.train_data.items()}
        if self.kg is None:
            datum = bert_preprocess(datum, self.max_seq_length, self.tokenizer)
        else:
            datum = kbert_preprocess(datum, self.max_seq_length, self.kg)
        # tree_datum = create_linear_semantic_tree(self.kg, datum, self.vocab)
        return datum

    def __len__(self):
        return len(self.train_data['text'])


class Causal_Test_Dataset(torch.utils.data.Dataset):
    '''
    @ Tian Li
    '''

    def __init__(self, test_data, kg, max_seq_length):
        self.test_data = test_data
        self.kg = kg
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __getitem__(self, i):
        datum = {key: value[i] for key, value in self.test_data.items()}
        if self.kg is None:
            datum = bert_preprocess(datum, self.max_seq_length, self.tokenizer)
        else:
            datum = kbert_preprocess(datum, self.max_seq_length, self.kg)
        # tree_datum = create_linear_semantic_tree(self.kg, datum, self.vocab)
        return datum

    def __len__(self):
        return len(self.test_data['text'])


class Causal_Dataset(object):
    '''
    @Tian Li
    '''

    def __init__(self, args, data_reader, graph_path=None, vocab=None, predicate=True):
        self.train_data, self.dev_data = data_reader.read_data()
        self.vocab = vocab
        if args.use_kg:
            self.kg = KnowledgeGraph(args, graph_path, predicate=predicate, vocab=self.vocab)
        else:
            self.kg = None
        self.args = args
        # self.kg=None

    def split(self):
        pos_idx = np.where(self.dev_data['label'] > 0.)[0]
        neg_idx = np.where(self.dev_data['label'] == 0.)[0]

        dev_idx = np.concatenate(
            [pos_idx[0:len(pos_idx) // 2], neg_idx[0:len(neg_idx) // 2]],
            axis=0
        ).astype(np.int32)  # dev_idx is now discarded
        test_idx = np.concatenate(
            [pos_idx[len(pos_idx) // 2:], neg_idx[len(neg_idx) // 2:]],
            axis=0
        ).astype(np.int32)

        dev_data = {key: [value[i] for i in dev_idx] for key, value in self.dev_data.items()}
        test_data = {key: [value[i] for i in test_idx] for key, value in self.dev_data.items()}
        return Causal_Train_Dataset(self.train_data, self.kg, self.args.seq_length), \
        Causal_Test_Dataset(dev_data, self.kg,self.args.seq_length), Causal_Test_Dataset(test_data, self.kg,self.args.seq_length)


class DA_train_dataset(torch.utils.data.Dataset):
    # incomplete
    def __init__(self, labeled_data, unlabeled_data, max_seq_length, kg):
        super(DA_train_dataset, self).__init__()
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data
        self.len_labeled = len(self.labeled_data['text'])
        self.len_unlabeled = len(self.unlabeled_data['text'])
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.kg = kg
        # self.augmenter = augmenter

    def __len__(self):
        return max(self.len_labeled, self.len_unlabeled)
        # return self.len_labeled

    def __getitem__(self, i):
        assert self.len_labeled<self.len_unlabeled
        l_ind = i*self.len_labeled//self.len_unlabeled
        # l_ind = i

        labeled_datum = {k: self.labeled_data[k][l_ind] for k in self.labeled_data.keys()}
        if self.kg is None:
            labeled_datum = bert_preprocess(labeled_datum, self.max_seq_length, self.tokenizer)
        else:
            labeled_datum = kbert_preprocess(labeled_datum, self.max_seq_length, self.kg)

        unlabeled_datum = {k: self.unlabeled_data[k][i] for k in self.unlabeled_data.keys()}
        if self.kg is None:
            unlabeled_datum = bert_preprocess(unlabeled_datum, self.max_seq_length, self.tokenizer)
        else:
            unlabeled_datum = kbert_preprocess(unlabeled_datum, self.max_seq_length, self.kg)
        return labeled_datum, unlabeled_datum


class DA_test_dataset(torch.utils.data.Dataset):
    # incomplete
    def __init__(self, labeled_data, max_seq_length, kg):
        super(DA_test_dataset, self).__init__()
        self.labeled_data = labeled_data
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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

    def __init__(self, args, source_reader, target_reader, graph_path=None, predicate=True, use_custom_vocab=None):
        super(DA_Dataset, self).__init__()
        self.source_data = source_reader.read_data()  # return a dict {'labeled':data, 'unlabeled':data}
        self.target_data = target_reader.read_data()
        self.max_seq_length = args.seq_length
        # self.augmenter = augment_factory[args.augmenter](args)
        self.vocab = None
        if args.use_kg:
            self.kg = KnowledgeGraph(args, graph_path, predicate=predicate, vocab=self.vocab)
        else:
            self.kg = None

    def split(self):
        labeled_src = self.source_data['labeled']
        keys = self.source_data['unlabeled'].keys()
        unlabeled = {k: self.source_data['unlabeled'][k] + self.target_data['unlabeled'][k] for k in keys}
        labeled_tgt = self.target_data['labeled']

        len_dev = len(labeled_src['text'])
        inds = list(range(len_dev))
        # random.shuffle(inds)
        labeled_src = {k:[labeled_src[k][i] for i in inds] for k in labeled_src.keys()}
        dev_data = {k:labeled_src[k][len_dev//5*4:] for k in labeled_src.keys()}
        train_labeled = {k:labeled_src[k][:len_dev//5*4] for k in labeled_src.keys()}
        return DA_train_dataset(train_labeled, unlabeled, self.max_seq_length, self.kg), \
                DA_test_dataset(dev_data, self.max_seq_length, self.kg), \
                DA_test_dataset(labeled_tgt, self.max_seq_length, self.kg)


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
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return self.length 

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


class DA_SSL_dataset(torch.utils.data.Dataset):
    def __init__(self, args, source_reader, target_reader, graph_path, memory_bank, predicate=True):
        super(DA_SSL_dataset,self).__init__()
        self.source_data = source_reader.read_data()
        self.target_data = target_reader.read_data()
        self.max_seq_length = args.seq_length
        self.memory_bank = memory_bank

        memory_bank.initialize(self.source_data, self.target_data)
        if args.use_kg:
            self.kg = KnowledgeGraph(args, graph_path, predicate=predicate, vocab=None)
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



num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


def create_vocab(sentence_list, vocab_size=10000):
    '''
    sentence_list: tokenized sentence list
    '''
    print('Creating vocab ...')

    total_tokens, unique_tokens = 0, 0
    token_freqs = {}

    for sent in sentence_list:
        # words = line.split()
        for token in sent:
            # if skip_len > 0 and len(words) > skip_len:
            #     continue

            # for w in words:
            if not bool(num_regex.match(token)):
                try:
                    token_freqs[token] += 1
                except KeyError:
                    unique_tokens += 1
                    token_freqs[token] = 1
                total_tokens += 1
        # fin.close()

    print('  %i total tokens, %i unique tokens' % (total_tokens, unique_tokens))
    sorted_token_freqs = sorted(token_freqs.items(), key=operator.itemgetter(1), reverse=True)
    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    index = len(vocab)
    for token, _ in sorted_token_freqs:
        vocab[token] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    print(' keep the top %i words' % vocab_size)

    return vocab


dataset_factory = {'causal_inference': Causal_Dataset,
                   'sentim': Causal_Dataset,
                   'domain_adaptation': DA_Dataset,
                   'DA_SSL':DA_SSL_dataset}

if __name__ == '__main__':
    from utils.readers import reader_factory
    # from utils.vocab import Vocab
    from utils.constants import *
    from collate_fn import collate_factory_train, collate_factory_eval
    from memory_bank import MemoryBank
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--vocab_path", default=None, type=str)
    parser.add_argument('--seq_length', default=256, type=int)
    parser.add_argument('--augmenter', default='synonym_substitution')
    parser.add_argument('--aug_rate', default=0.7, help='aug_rate for synonym_substitution')
    parser.add_argument('--use_kg', action='store_true')
    parser.add_argument('--model', default='base_DA', type=str)
    parser.add_argument('--pos_require_knowledge', type=str, help='the part of speech that \
        requires kg to add knowledge, choose a subset from [ADJ, ADP, ADV, CONJ, DET, NOUN, \
        NUM, PRT, PRON, VERB, ., X], split with "," e.g. ADJ,ADP,ADV', default='ADJ,ADV')
    parser.add_argument('--use_pivot_kg', action='store_true')
    parser.add_argument('--num_pivots', type=int, default=2000)
    parser.add_argument('--min_occur', type=int, default=5)
    parser.add_argument('--update_steps', type=int, default=10)
    parser.add_argument('--dataset')
    parser.add_argument('--task')
    parser.add_argument('--source')
    parser.add_argument('--target')

    # parser.add_argument('--')

    args = parser.parse_args()
    if args.use_pivot_kg:
        args.vocab_require_knowledge = load_pivots(args)
    else:
        args.vocab_require_knowledge = None
    args.kg_path = ['data/imdb_sub_conceptnet_new.spo']
    args.pollution_rate = [0.7,0.9]

    if args.task == 'domain_adaptation' or args.task=='DA_SSL':
        source = args.source
        if '.' in source:
            # print('source')
            lst = source.split('.')
            dataset_name = lst[0]
            domain_name = lst[1]
            source_reader = reader_factory[dataset_name](domain_name, 'source')
        else:
            source_reader = reader_factory[args.source]()

        target = args.target
        if '.' in target:
            lst = target.split('.')
            dataset_name = lst[0]
            domain_name = lst[1]
            target_reader = reader_factory[dataset_name](domain_name, 'target')
        else:
            target_reader = reader_factory[args.target]()

        if args.task == 'DA_SSL':
            memory_bank = MemoryBank(args)
            dataset = dataset_factory[args.task](args, source_reader, target_reader, graph_path=args.kg_path, memory_bank=memory_bank)
        else:
            dataset = dataset_factory[args.task](args, source_reader, target_reader, graph_path=args.kg_path)
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

    # vocab = Vocab()
    # vocab.load(args.vocab_path)
    # args.vocab = vocab

    # source_reader = reader_factory['bdek']('books','source')
    # target_reader = reader_factory['bdek']('kitchen','target')

    # dataset = dataset_factory['domain_adaptation'](args, source_reader, target_reader, graph_path=['data/imdb_sub_conceptnet_new.spo'])
    # # print(dataset.target_data['labeled']['text'][0])
    # train_dataset, dev_dataset, eval_dataset = dataset.split()
    # data_reader = reader_factory['imdb']([0.9, 0.7], False)
    # dataset = dataset_factory['causal_inference'](args, data_reader, graph_path=['data/imdb_sub_conceptnet.spo'])
    # train_dataset, dev_dataset, eval_dataset = dataset.split()
    collate_fn_eval = collate_factory_eval[args.model]
    collate_fn_train = collate_factory_train[args.model]
    # train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # dev_sampler = torch.utils.data.RandomSampler(dev_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn_train)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=8, num_workers=0, collate_fn=collate_fn_eval)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, num_workers=0, collate_fn=collate_fn_eval)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokens = tokenizer.encode('I love Peking University. Describe the city you live in.')
    # print(tokens)
    # nltk.download('punkt')
    from transformers import BertTokenizer
    t = BertTokenizer.from_pretrained('bert-base-uncased')
    for i, (labeled_batch,_,_) in enumerate(train_loader):
        print(labeled_batch.keys())
        print(labeled_batch['tokens_org'][0])
        # print(labeled_batch['tokens_kg'][0])
        # print(labeled_batch['pos'][0])
        # print(labeled_batch['vm'][0])
        print(labeled_batch['text'][0])
        print(labeled_batch['label'])
        print(t.decode(labeled_batch['tokens_org'][0]))
        print(labeled_batch['ssl_label'][0])
        ssl_label = (labeled_batch['ssl_label'][0]!=-1) * labeled_batch['ssl_label'][0]
        print(t.decode(ssl_label))
        # print(t.decode(labeled_batch['tokens_kg'][0]))
        # print(labeled_batch['tokens'][0])
        # print(labeled_batch['pos'][0])
        # print(labeled_batch['vm'][0][:16,:16])
        # print(labeled_batch['text'][0])
        # print(t.decode(labeled_batch['tokens'][0]))
        # print(labeled_batch['aug_tokens'][0])
        # print(unlabeled_src_batch['domain'])
        # print(unlabeled_tgt_batch['domain'])
        # print(source_batch['domain'])
        # print(target_batch['domain'])
        # print(labeled_batch['aug_tokens'][0])
        # print(labeled_batch['label'][0])
        # assert labeled_batch['text'][0]!=labeled_batch['aug_text'][0]
        print(i)
        # print(unlabeled_src_batch['text'][0])
        # print(unlabeled_src_batch['aug_text'][0])
        # print(unlabeled_src_batch['domain'][0])
        # print(unlabeled_batch['tokens'][0])
        # print(unlabeled_batch['aug_tokens'][0])
        # assert unlabeled_batch['text'][0]!=unlabeled_batch['aug_text'][0]
        # print(labeled_batch['mask'])
        # print(labeled_batch['label'][0])
        if i==1:
            break
