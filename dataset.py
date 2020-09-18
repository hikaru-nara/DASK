import os, numpy as np
from os.path import join
import random
from uer.utils.constants import *
# import scipy.sparse as sp
# from gensim.corpora import Dictionary as gensim_dico
from brain.knowgraph import KnowledgeGraph
from brain.config import *
import operator
import re
import xml.etree.ElementTree as ET

import torch

dataset_factory = {'domain_adaptation':DA_Dataset,
            'causal_inference':Causal_Dataset}

def create_linear_semantic_tree(kg, datum, vocab):
    # get semantic tree (i.e. token_list), position embedding and visible matrices 
    token_list, position_list, visible_matrix = kg.add_knowledge_with_vm(datum['text'])
    token_ids = np.array([vocab.get(t, UNK_ID) for t in token_list])
    mask = np.array([1 if t != PAD_TOKEN else 0 for t in token_list])
    position_array = np.array(position_list)
    datum['token'], datum['mask'], datum['pos'], datum['vm'] = token_ids, mask, position_array, visible_matrix
    return datum

class Causal_Dataset(objest):
    '''
    @Tian Li
    '''
    def 

class DA_Train_Dataset(torch.utils.data.IterableDataset):
    #incomplete
    def __init__(self, labeled_data, unlabeled_data, kg, repeat):
        super(DA_Train_Dataset, self).__init__()
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data
        self.repeat = repeat
        self.len_labeled_data = len(self.labeled_data)
        self.len_unlabeled_data = len(self.unlabeled_data)
        self.length = max(self.len_labeled_data, self.len_unlabeled_data)*self.repeat
        self.kg = kg

    def __len__(self):
        return self.length

    
    def __iter__(self):
        num = 0
        worker_info = torch.utils.data.get_worker_info()
        # single process
        if worker_info is None:
            rg = default_rng()
            while num<self.length:
                labeled_ids = math.floor(rg.uniform()*self.len_labeled_data)
                unlabeled_ids = math.floor(rg.uniform()*self.len_unlabeled_data)
                labeled_datum = {key:data[labeled_ids] for key,data in self.labeled_data.items()}
                unlabeled_datum =  {key:data[unlabeled_ids] for key,data in self.unlabeled_data.items()}
                labeled_datum = create_linear_semantic_tree(labeled_datum)
                unlabeled_datum = create_linear_semantic_tree(unlabeled_datum)
                yield labeled_datum, unlabeled_datum
                num += 1
        else:
            # set different seed for different workers 
            # refer to randomness in multiprocess dataloading in the pytorch doc below
            # https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader
            rg = default_rng(seed=torch.initial_seed())
            num_workers = worked_info.num_workers
            total_iterations = self.length // num_workers
            while num<total_iterations:
                labeled_ids = math.floor(rg.uniform()*self.len_labeled_data)
                unlabeled_ids = math.floor(rg.uniform()*self.len_unlabeled_data)
                labeled_datum = {key:data[labeled_ids] for key,data in self.labeled_data.items()}
                unlabeled_datum =  {key:data[unlabeled_ids] for key,data in self.unlabeled_data.items()}
                labeled_datum = create_linear_semantic_tree(labeled_datum)
                unlabeled_datum = create_linear_semantic_tree(unlabeled_datum)
                yield labeled_datum, unlabeled_datum
                num += 1


class DA_Validation_Dataset(torch.utils.data.Dataset):
    #incomplete
    def __init__(self, labeled_data):
        self.labeled_data = labeled_data

    def __len__(self):
        return len(self.labeled_data)

    def __getitem__(self, i):
        return self.labeled_data[i]



class DA_Dataset(object):
    def __init__(self, source_reader, target_reader, graph_path, predicate=True, \
                repeat=1, use_custom_vocab=False):
        super(dataset, self).__init__()

        self.source_data = source_reader.read_data() # return a dict {'labeled':data, 'unlabeled':data}
        self.target_data = target_reader.read_data()
        self.vocab = create_vocab(self.source_data['labeled']['text']+self.source_data['unlabeled']['text'] \
                                +self.target_data['labeled']['text']+self.target_data['unlabeled']['text'])
        self.kg = KnowledgeGraph(graph_path, predicate=predicate, use_custom_vocab=use_custom_vocab, vocab=self.vocab)
        self.repeat = repeat

    def split(self, args):
        supervision_rate = args.supervision_rate
        semi_sup_data, val_data = split_semi_supervision(self.target_data['labeled'], supervision_rate)
        labeled_train_data = {key: val1+val2 for key, val1, val2 in zip(semi_sup_data.keys(), \
                                    semi_sup_data.values(), self.source_data['labeled'].values())}
        unlabeled_train_data = {key: val1+val2 for key, val1, val2 in zip(self.target_data['unlabeled'].keys(), \
                                    self.target_data['unlabeled'].values(), self.source_data['unlabeled'].values())}
        return DA_Train_Dataset(labeled_train_data, unlabeled_train_data), DA_Validation_Dataset(val_data)



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

    print ('  %i total tokens, %i unique tokens' % (total_tokens, unique_tokens))
    sorted_token_freqs = sorted(token_freqs.items(), key=operator.itemgetter(1), reverse=True)
    vocab = {'<pad>':0, '<unk>':1, '<num>':2}
    index = len(vocab)
    for token, _ in sorted_token_freqs:
        vocab[token] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    print (' keep the top %i words' % vocab_size)
    
    return vocab


if __name__=='__main__':
    dataset = dataset('books','dvd',graph_path=['brain/kgs/conceptnet-assertions-5.7.0.csv'])
    print(dataset.length_histogram)

