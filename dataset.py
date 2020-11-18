import os, numpy as np
from os.path import join
import random
# import scipy.sparse as sp
# from gensim.corpora import Dictionary as gensim_dico
from brain.knowgraph import KnowledgeGraph
# from brain.config import *
from utils.utils import standardize
import operator
import re
# import xml.etree.ElementTree as ET
# from wrapper_tokenizer import gpt2_tokenizer
from transformers import BertTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import math
import torch
from numpy.random import default_rng
from uer.utils.constants import *
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
    # if len(tokens) > max_seq_length:
    #     tokens = tokens[:max_seq_length - 1] + [tokens[-1]]

    datum['tokens'] = tokens
    datum['mask'] = [1 for _ in range(len(datum['tokens']))]
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
    token_list, position_list, visible_matrix, _ = kg.add_knowledge_with_vm(datum['text'], max_length=max_seq_length)
    token_list = [CLS_ID] + token_list[:-1]
    mask = np.array([1 if t != PAD_TOKEN else 0 for t in token_list])
    datum['tokens'] = np.array(token_list)
    datum['pos'] = np.array(position_list)
    datum['vm'] = visible_matrix
    datum['mask'] = mask
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
        # datum['text'] = 'I love peking university'
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
            self.kg = KnowledgeGraph(graph_path, predicate=predicate, vocab=self.vocab)
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

        test_data = {key: [value[i] for i in test_idx] for key, value in self.dev_data.items()}
        return Causal_Train_Dataset(self.train_data, self.kg, self.args.seq_length), \
        Causal_Test_Dataset(test_data, self.kg, self.args.seq_length)



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

    def __getitem__(self, i):
        assert self.len_labeled<self.len_unlabeled
        l_ind = i*self.len_labeled/self.len_unlabeled

        labeled_datum = {k: self.labeled_data[k][l_ind] for k in self.labeled_data.keys()}
        if self.kg is None:
            labeled_datum = bert_preprocess(labeled_datum, self.max_seq_length, self.tokenizer)
        else:
            labeled_datum = kbert_preprocess(labeled_datum, self.max_seq_length, self.kg)

        unlabeled_datum = {k: self.unlabeled_data[k][i] for k in self.unabeled_data.keys()}
        if self.kg is None:
            unlabeled_datum = bert_preprocess(lunabeled_datum, self.max_seq_length, self.tokenizer)
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

    def __init__(self, args, source_reader, target_reader, graph_path=None, predicate=None, use_custom_vocab=None):
        super(DA_Dataset, self).__init__()
        self.source_data = source_reader.read_data()  # return a dict {'labeled':data, 'unlabeled':data}
        self.target_data = target_reader.read_data()
        self.max_seq_length = args.seq_length
        # self.augmenter = augment_factory[args.augmenter](args)
        self.rng = default_rng()

    def split(self):
        labeled_src = self.source_data['labeled']
        keys = self.source_data['unlabeled'].keys()
        unlabeled = {k: self.source_data['unlabeled'][k] + self.target_data['unlabeled'][k] for k in keys}
        labeled_tgt = self.target_data['labeled']

        len_dev = len(labeled_src['text'])
        dev_data = {k:labeled_src[k][:len_dev//5] for k in labeled_src['text']}
        train_labeled = {k:labeled_src[k][len_dev//5:] for k in labeled_src['text']}
        return DA_train_dataset(train_labeled, unlabeled, self.max_seq_length), \
                DA_test_dataset(dev_data, self.max_seq_length), \
                DA_test_dataset(labeled_tgt, self.max_seq_length)



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
                   'base_DA': DA_Dataset}

if __name__ == '__main__':
    from utils.readers import reader_factory
    # from utils.vocab import Vocab
    from utils.constants import *

    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--vocab_path", default=None, type=str)
    parser.add_argument('--seq_length', default=256, type=int)
    parser.add_argument('--augmenter', default='synonym_substitution')
    parser.add_argument('--aug_rate', default=0.7, help='aug_rate for synonym_substitution')
    parser.add_argument('--use_kg', action='store_true')

    # parser.add_argument('--')

    args = parser.parse_args()

    # vocab = Vocab()
    # vocab.load(args.vocab_path)
    # args.vocab = vocab

    source_reader = reader_factory['imdb']([0.9,0.7])
    # target_reader = reader_factory['bdek']('kitchen','target')
    dataset = dataset_factory['sentim'](args, source_reader, graph_path=['data/imdb_sub_conceptnet.spo'])
    train_dataset, eval_dataset = dataset.split()
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # dev_sampler = torch.utils.data.RandomSampler(dev_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, \
                                               sampler=train_sampler)
    # dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=8, num_workers=2, collate_fn=collate_fn_SSL_dev, sampler=dev_sampler)
    # eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8, num_workers=2, collate_fn=collate_fn_SSL_eval)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokens = tokenizer.encode('I love Peking University. Describe the city you live in.')
    # print(tokens)
    # nltk.download('punkt')
    from transformers import BertTokenizer
    t = BertTokenizer.from_pretrained('bert-base-uncased')
    for i, labeled_batch in enumerate(train_loader):
        print(labeled_batch['tokens'][0])
        print(labeled_batch['pos'][0])
        print(labeled_batch['vm'][0])
        print(labeled_batch['text'][0])
        print(t.decode(labeled_batch['tokens'][0]))
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
        if i==10:
            break
