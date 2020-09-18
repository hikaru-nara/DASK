import os, numpy as np
from os.path import join
import random
# import scipy.sparse as sp
# from gensim.corpora import Dictionary as gensim_dico
from brain.knowgraph import KnowledgeGraph
from brain.config import *
import operator
import re
import xml.etree.ElementTree as ET

import torch

reader_factory = {'bdek':bdek_reader}



class bdek_reader(object):
    def __init__(self, domain_name, source_or_target):
        '''
        class for read raw bdek data from disk; 
        domain_name in 'books', 'dvd', 'kitchen', 'electronics'; 
        source_or_target in 'source' or 'target'

        pass an obj of this class to a da_dataset object defined in ../dataset.py
        '''
        super(bdek_dataset, self).__init__()
        self.text_paths = {
                    'positive':join('data/amazon-review-old',domain_name,'positive.parsed'),\
                    'negative':join('data/amazon-review-old',domain_name,'negative.parsed'),\
                    'unlabeled':join('data/amazon-review-old',domain_name,'{}UN.txt'.format(source_name)),\
                    }

        self.graph_feature_paths = {
                            'labeled':'graph_features/sf_' + domain_name +'_small_5000.np', \
                            'unlabeled':'graph_features/sf_' + domain_name +'_test_5000.np', \
                            }
        assert source_or_target=='source' or source_or_target=='target'
        self.domain_label = int(source_or_target=='target')

    def read_data(self):
        '''
        major read data procedure; called from da_dataset
        '''
        labeled_data = {}
        positive_text = self.get_dataset(self.text_paths['positive'])
        negative_text = self.get_dataset(self.text_paths['negative'])
        positive_label = [1]*len(positive_text)
        negative_label = [0]*len(negative_text)

        labeled_data['text'] = positive_text + negative_text
        labeled_data['label'] = positive_label + negative_label
        labeled_data['domain'] = [self.domain_label] * len(labeled_data['text'])
        labeled_data['graph'] = np.load(open(self.graph_feature_paths['labeled'], 'rb'), allow_pickle=True)

        unlabeled_data['text'] = self.get_dataset(self.text_paths['unlabeled'])
        unlabeled_data['domain'] = [self.domain_label] * len(unlabeled_data['text'])
        unlabeled_data['graph'] = np.load(open(self.graph_feature_paths['unlabeled'], 'rb'), allow_pickle=True)

        return labeled_data, unlabeled_data

    def get_dataset(self, file_path):
        '''
        extract texts from xml format file; see data/books/positive.parsed for an instinct of the format
        return a list of sentences, where each sentence is a list of words (may contain multiple lines)
        '''
        tree = ET.parse(file_path)
        root = tree.getroot()
        sentences = []
        for review in root.iter('review'):
            sentences.append(review)
        return sentences






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
    dataset = bdek_dataset('books','dvd',graph_path=['brain/kgs/conceptnet-assertions-5.7.0.csv'])
    print(dataset.length_histogram)

