# coding: utf-8
"""
KnowledgeGraph
"""
import os
import sys
sys.path.append('/rscratch/tianli/DA_NLP/graph-causal-domain-adaptation')
import brain.config as config
# import fairseq.data.encoder.gpt2_bpe as gpt2_bpe
# import wrapper_tokenizer as wt
import numpy as np
from utils.utils import standardize
from uer.utils.constants import *
from transformers import BertTokenizer
import re
import nltk
from nltk.tokenize import word_tokenize
from utils.filters import filter_factory
nltk.download('universal_tagset')
class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, args, spo_files, use_custom_vocab=False, vocab=None, predicate=True, memory_bank=None):
        self.predicate = predicate
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        # self.lookup_table = {}
        graphfilter = filter_factory[args.filter](args)
        if graphfilter is not None:
            self.spo_file_paths = graphfilter.filter(self.spo_file_paths)
        else:
            print('warning!!no filter')
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        # if use_custom_vocab:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # else:
        #     self.tokenizer = wt.gpt2_tokenizer()
        self.special_tags = set(config.NEVER_SPLIT_TAG)
        self.add_kg_pos = args.pos_require_knowledge.split(',')
        self.memory_bank = memory_bank
        if args.vocab_require_knowledge is not None:
            self.add_kg_vocab = args.vocab_require_knowledge['word']
        else:
            self.add_kg_vocab = None
        for pos in self.add_kg_pos:
            assert pos in config.POS_SET
        print('max entities', config.MAX_ENTITIES)

    def _create_lookup_table(self):
        # lookup_table = {}
        # for spo_path in self.spo_file_paths:
        #     print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
        #     with open(spo_path, 'r', encoding='utf-8') as f:
        #         for line in f:
        #             # print('line')
        #             # print(line)
        #             # print(line.split('\t'))
        #             try:
        #                 lst = line.split('\t')

        #                 subj, pred, obje = lst[0], lst[1], lst[2][:-1]
        #                 # subj = subj[:subj.rfind('/')]
        #                 # subj = subj[subj.rfind('/')+1:]
        #                 # pred = pred[pred.rfind('/')+1:]
        #                 # obje = obje[pred.rfind('/')+1:]
        #             except:
        #                 print("[KnowledgeGraph] Bad spo:", line)
        #             if self.predicate:

        #                 pred = '_'.join(re.findall('[A-Z][^A-Z]*', pred)) # RelatedTo -> Related_To
        #                 value = pred + '_' + obje
        #             else:
        #                 value = obje
        #             if subj in lookup_table.keys():
        #                 lookup_table[subj].add(value)
        #             else:
        #                 lookup_table[subj] = set([value])
        # return lookup_table
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        lst = line.split('\t')

                        subj, pred, obje = lst[0], lst[1], lst[2][:-1]
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    # print(pred)
                    # pred = '_'.join(re.findall('[A-Z][^A-Z]*', pred)) # RelatedTo -> Related_To
                    # print(pred)
                    value = pred + '_' + obje                    
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])

                    value = subj + '_' + pred
                    if obje in lookup_table.keys():
                        lookup_table[obje].add(value)
                    else:
                        lookup_table[obje] = set([value])

                    value = subj + '_' + obje
                    if pred in lookup_table.keys():
                        lookup_table[pred].add(value)
                    else:
                        lookup_table[pred]=set([value])
                    # if self.predicate:

                    #     pred = '_'.join(re.findall('[A-Z][^A-Z]*', pred)) # RelatedTo -> Related_To
                    #     value = pred + '_' + obje
                    # else:
                    #     value = obje
                    # if subj in lookup_table.keys():
                    #     lookup_table[subj].add(value)
                    # else:
                    #     lookup_table[subj] = set([value])
        return lookup_table

    # def add_knowledge_with_vm_new(self, sentence, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=256, add_special_tokens=True):
        


    def add_knowledge_with_vm(self, sentence, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=256, add_special_tokens=True):
        '''
        need to tune max_length on bdek dataset; draw the histgram of length and decide a threshold
        '''
        # 加的东西太多-》
        # 冠词、介词不要
        # 形容词、副词和常规动词，和少量名次 加 kg
        # 加很多个节点，做embedding的平均； bert pretrained embedding，先存起来；
        # multi stage
        sent_tree = []
        pos_idx_tree = []
        abs_idx_tree = []
        pos_idx = -1
        abs_idx = -1
        abs_idx_src = []
        
        # split_sent = sentence.split(' ')
        split_sent = word_tokenize(sentence)

        pos_tag = nltk.pos_tag(split_sent, tagset='universal')
        if add_special_tokens:
            pos_tag = [(CLS_TOKEN, 'X')] + pos_tag
            split_sent = [CLS_TOKEN] + split_sent
        # pos_tag = nltk.pos_tag(split_sent)
        # split_sent = [standardize(token) for token in split_sent]
        # print(split_sent[0])
        for idx, token in enumerate(split_sent):
            # print(max_entities)
            # print(token, pos_tag[idx])
            # print(self.add_kg_pos)
            self.add_kg_vocab = self.memory_bank.pivots
            if self.add_kg_vocab is None:
                if pos_tag[idx][1] in self.add_kg_pos and token != 'i':
                    # print('add knowledge')
                    entities = list(self.lookup_table.get(token, []))[:max_entities]
                else:
                    entities = []
            else:
                # print(token)
                # print(token in self.add_kg_vocab)
                if token in self.add_kg_vocab:
                    
                    entities = list(self.lookup_table.get(token, []))[:max_entities]

                else:
                    entities = []
                # print(len(entities))
            # print(token, entities)
            token = self.tokenizer.encode(token.strip(), add_special_tokens=False)
            if len(token)==0:
                continue
            entities = [' '.join(ent.split('_')) for ent in entities]
            entities = [self.tokenizer.encode(ent.strip(), add_special_tokens=False) for ent in entities]
            # print(token)
            # print(entities)
            sent_tree.append((token, entities))
            # token_pos_idx = [pos_idx+1]
            # token_abs_idx = [abs_idx+1]
            if not isinstance(token, list) and token in self.special_tags:
                token_pos_idx = [pos_idx+1]
                token_abs_idx = [abs_idx+1]
            else:
                # token = token.strip().split(' ')
                # token = self.tokenizer.encode(token.strip())
                token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)]
                token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
            # if yesprint:
            # print('add kg')
            # print('[{}]'.format(tmp), token, token_abs_idx)
            abs_idx = token_abs_idx[-1]

            entities_pos_idx = []
            entities_abs_idx = []
            for ent in entities:
                # ent = ent.split('_')
                ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent)+1)]
                entities_pos_idx.append(ent_pos_idx)
                ent_abs_idx = [abs_idx + i for i in range(1, len(ent)+1)]
                abs_idx = ent_abs_idx[-1]
                entities_abs_idx.append(ent_abs_idx)

            pos_idx_tree.append((token_pos_idx, entities_pos_idx))
            pos_idx = token_pos_idx[-1]
            abs_idx_tree.append((token_abs_idx, entities_abs_idx))
            abs_idx_src += token_abs_idx

        # Get know_sent and pos
        know_sent = []
        pos = []
        seg = []
        for i in range(len(sent_tree)):
            word = sent_tree[i][0]
            if not isinstance(word, list) and word in self.special_tags:
                know_sent += [word]
                seg += [0] # 1？？
            else:
                # add_word = word.strip().split(' ')
                # add_word = self.tokenizer.encode(word.strip())
                add_word = list(word)
                know_sent += add_word 
                seg += [0] * len(add_word) # 1？？
            pos += pos_idx_tree[i][0]
            for j in range(len(sent_tree[i][1])):
                add_word = sent_tree[i][1][j]
                # add_word = add_word.strip().split(' ')
                # add_word = self.tokenizer.encode(word.strip())
                know_sent += add_word
                seg += [1] * len(add_word)
                pos += list(pos_idx_tree[i][1][j])

        

        token_num = len(know_sent)

        # Calculate visible matrix
        visible_matrix = np.zeros((token_num, token_num))
        for item in abs_idx_tree:
            src_ids = item[0]
            for id in src_ids:
                visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                visible_matrix[id, visible_abs_idx] = 1
            for ent in item[1]:
                for id in ent:
                    visible_abs_idx = ent + src_ids
                    visible_matrix[id, visible_abs_idx] = 1

        src_length = len(know_sent)

        if len(know_sent) < max_length:
            pad_num = max_length - src_length
            know_sent += [PAD_ID] * pad_num
            seg += [0] * pad_num
            pos += list(range(src_length, max_length))
            visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
        else:
            know_sent = know_sent[:max_length]
            seg = seg[:max_length]
            pos = pos[:max_length]
            visible_matrix = visible_matrix[:max_length, :max_length]
        
        # if len(know_sent) < max_length - 2:
        #     pad_num = max_length - src_length
        #     know_sent = [CLS_ID] + know_sent + [SEP_ID]
        #     know_sent += [PAD_ID] * (pad_num-2)
        #     pos = [0] + [p+1 for p in pos]
        #     pos += [max_length - 1] * (pad_num-1)
        #     visible_matrix = np.pad(visible_matrix, ((1,1),(1,1)), 'constant')
        #     visible_matrix[0,0] = 1 
        #     visible_matrix[-1,-1] = 1
        #     visible_matrix = np.pad(visible_matrix, ((0, pad_num-2), (0, pad_num-2)), 'constant')  # pad 0
            
        #     seg += [0] * pad_num
        # else:
        #     # if add_special_tokens:
        #     #     know_sent = [CLS_ID] + know_sent[:max_src_length] + [SEP_ID]
        #     know_sent = [CLS_ID] + know_sent[:max_length-2] + [SEP_ID]
        #     seg = seg[:max_length-2]
        #     pos = [0]+pos[:max_length-2]+[max_length-1]
        #     visible_matrix = visible_matrix[:max_length-2, :max_length-2]
        #     visible_matrix = np.pad(visible_matrix, ((1,1),(1,1)), 'constant')
        #     visible_matrix[0,0] = 1 
        #     visible_matrix[-1,-1] = 1
        
        return know_sent, pos, visible_matrix, seg


    def add_knowledge_with_vm_batch(self, sent_batch, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=256):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        for split_sent in split_sent_batch:

            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            for token in split_sent:

                entities = list(self.lookup_table.get(token, []))[:max_entities]
                sent_tree.append((token, entities))

                if token in self.special_tags:
                    token_pos_idx = [pos_idx+1]
                    token_abs_idx = [abs_idx+1]
                else:
                    token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)]
                    token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent)+1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent)+1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                else:
                    add_word = list(word)
                    know_sent += add_word 
                    seg += [0] * len(add_word)
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = list(sent_tree[i][1][j])
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
            
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)
        
        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch


if __name__ == '__main__':
    graph_path = 'data/bdek_sub_conceptnet.spo'
    kg = KnowledgeGraph([graph_path], predicate=True, use_custom_vocab=False, vocab=None)
    sentence = 'I love Peking Univerisity.'
    kg.add_knowledge_with_vm(sentence)
