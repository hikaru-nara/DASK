import os
import os.path as osp
import sys
import random
import copy
import numpy as np 
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils.filters import filter_factory
from .config import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class Triplet():
	def __init__(self, triplet):
		self.triplet = triplet
		self.pre_enc = True # TODO
		self.tokenizer = tokenizer
		if self.pre_enc:
			self.enc = [self.tokenizer.encode(t, add_special_tokens=False) for t in self.triplet]

	def activate(self):
		if not self.pre_enc:
			self.enc = [self.tokenizer.encode(t, add_special_tokens=False) for t in self.triplet]			
		# compare the speed of pre-encoding and encoding on-the-fly

class BranchBundle():
	def __init__(self, root, branches=[]):
		self.tokenizer = tokenizer
		self.root = root 
		self.branches = branches
		self.root_enc = tokenizer.encode(self.root, add_special_tokens=False)
		self.root_len = len(self.root_enc)
		for i in range(len(self.branches)):
			self.branches[i].activate()

	def tokenize(self):
		self.root_start = 0
		encoding = self.root_enc
		for b in self.branches:
			if self.root==b.triplet[0]:
				encoding += b.enc[1] + b.enc[2]
			elif self.root==b.triplet[1]:
				encoding = b.enc[0] + encoding + b.enc[2]
				self.root_start += len(b.enc[0])
			elif self.root==b.triplet[2]:
				encoding = b.enc[0] + b.enc[1] + encoding
				self.root_start += len(b.enc[0]+b.enc[1])
			else:
				print('Error: root is not in triplet!')
				exit()
		self.root_end = self.root_start + self.root_len
		self.enc_len = len(encoding)
		return encoding

	def get_pos_rel_to(self, start):
		pos = [start+i for i in range(self.root_len)]
		end = start + self.root_len
		for b in self.branches:
			if self.root==b.triplet[0]:
				pad_len_top = 0
				pad_len_bot = len(b.enc[1]) + len(b.enc[2])
			elif self.root==b.triplet[1]:
				pad_len_top = len(b.enc[0])
				pad_len_bot = len(b.enc[2])
			elif self.root==b.triplet[2]:
				pad_len_top = len(b.enc[0]) + len(b.enc[1])
				pad_len_bot = 0
			else:
				print('Error: root is not in triplet!')
				exit()

			pos_front = [i for i in range(start-pad_len_top, start)]
			pos_back = [i for i in range(end, end+pad_len_bot)]
			pos = pos_front + pos + pos_back

		return pos

	def get_vm_mask(self, start, max_length):
		mask = np.ones((self.root_len, self.root_len))
		root_st = 0
		for b in self.branches:
			cur_mask_len = mask.shape[0]
			only_root = [0. for _ in range(cur_mask_len)]
			only_root[root_st:root_st+self.root_len] = [1 for _ in range(self.root_len)]
			if self.root==b.triplet[0]:
				pad_len_top = 0
				pad_len_bot = len(b.enc[1]) + len(b.enc[2])
			elif self.root==b.triplet[1]:
				root_st += len(b.enc[0])
				pad_len_top = len(b.enc[0])
				pad_len_bot = len(b.enc[2])
			elif self.root==b.triplet[2]:
				root_st += len(b.enc[0]+b.enc[1])
				pad_len_top = len(b.enc[0]) + len(b.enc[1])
				pad_len_bot = 0
			else:
				print('Error: root is not in triplet!')
				exit()
			preserve_mask = np.ones_like(mask)
			preserve_mask = np.pad(preserve_mask, ((pad_len_top, pad_len_bot),\
					(pad_len_top, pad_len_bot)), constant_values=0.)
			mask = np.pad(mask, ((pad_len_top, pad_len_bot),\
					(pad_len_top, pad_len_bot)), constant_values=1.)
			branch_mask = np.array([[1. for _ in range(pad_len_top)] + \
				only_root \
				+[1. for _ in range(pad_len_bot)]])
			
			branch_mask = branch_mask.T.dot(branch_mask)
			branch_mask = np.logical_or(branch_mask,preserve_mask) # preserve structure in the original mask, only update the new-comers
			mask *= branch_mask

		global_mask = np.ones((max_length, max_length))
		global_mask[start:start+self.enc_len] = 0.
		global_mask[:, start:start+self.enc_len] = 0.
		global_mask[start+self.root_start: start+self.root_end] = 1.
		global_mask[:, start+self.root_start: start+self.root_end] = 1.
		global_mask[start:start+self.enc_len, start:start+self.enc_len] = mask 

		return global_mask
		

class SentenceTrunk():
	def __init__(self):
		self.bundles = []

	def push_back(self, bundle):
		self.bundles.append(bundle)

	def get_pos(self, max_length, debug=False):
		if debug:
			print('getpos')
		pos = []
		num_bun = 0
		
		for bun, pos_i in zip(self.bundles, self.start_in_trunk):
			bun_pos = bun.get_pos_rel_to(pos_i)
			if debug:
				print(bun.root, [b.triplet for b in bun.branches])
				print(bun_pos)
			if len(bun_pos+pos) >= max_length: # leave one out for SEP
				break 
			num_bun += bun.root_len
			pos += bun_pos
		pos += [num_bun+i for i in range(max_length-len(pos))]
		min_pos = min(pos) # if there is negative pos
		pos = [p-min_pos for p in pos]
		return np.array(pos)

	def flatten(self, max_length, debug=False):
		if debug:
			print('flatten')
		self.bundles_start = [-1 for _ in range(len(self.bundles))]
		self.start_in_trunk = [-1 for _ in range(len(self.bundles))]
		trunk_len = 0
		tokens = []
		pad_mask = []
		for i, bun in enumerate(self.bundles):
			bun_enc = bun.tokenize()
			if debug:
				print(bun.root, [b.triplet for b in bun.branches])
				print(bun_enc)
				print(tokenizer.decode(bun_enc))
			if len(bun_enc+tokens) >= max_length: # leave one out for SEP
				break 
			self.bundles_start[i] = len(tokens)
			self.start_in_trunk[i] = trunk_len
			trunk_len += bun.root_len
			tokens += bun_enc
			pad_mask += [1 for _ in range(len(bun_enc))]
		tokens += [PAD_ID for i in range(max_length-len(tokens))]
		non_padding_len = len(pad_mask)
		pad_mask += [0 for i in range(max_length-len(pad_mask))]
		return np.array(tokens), np.array(pad_mask), non_padding_len 

	def get_vm(self, max_length, debug=False):
		# please call flatten before get_vm, to compute self.bundles_start
		vm = np.ones((max_length,max_length))
		row = 0
		for bun, st in zip(self.bundles, self.bundles_start):
			
			if st == -1:
				break
			bun_mask = bun.get_vm_mask(st, max_length)
			vm *= bun_mask
			if debug:
				print(bun.root, [b.triplet for b in bun.branches])
				print(bun_mask)
		return vm



class KnowledgeGraph(object):
	def __init__(self, args, spo_files, memory_bank=None, **kwargs):
		self.debug = args.debug
		self.spo_file_paths = spo_files
		graph_filter = filter_factory[args.filter](args)
		if graph_filter is not None:
			print('Info: Filtering the graph')
			self.spo_file_paths = graph_filter.filter(self.spo_file_paths)
		else:
			print('Warning: No filter applied')

		print('Info: creating lookup_table')
		self.lookup_table = self.create_lookup_table(self.spo_file_paths)
		self.model_name = 'roberta'  if 'roberta' in args.model_name else 'bert'
		
		initialize_constant(self.model_name)
		self.memory_bank = memory_bank

		self.max_facts = 2 # args.max_facts # default to 2
		print('Info: KG complete')
		# self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	def create_lookup_table(self, file_paths):
		'''
		create a mapping from words to pivots
		'''
		lookup_table = {}
		for path in file_paths:
			with open(path, 'r') as f:
				for n,line in enumerate(f):
					try:
						lst = line.split('\t')
						head, pred, tail = lst[0], lst[1], lst[2]
						if tail[-1]=='\n':
							tail = tail[:-1]
					except:
						print('bad triplet at line {0}: {1}'.format(n,line))
						continue
					lst = [head, pred, tail]
					for i, p in enumerate(lst):
						if p in lookup_table.keys():
							lookup_table[p].append(Triplet(lst))
						else:
							lookup_table[p] = [Triplet(lst)]
		return lookup_table

	def process_word_list(self,sentence, word_list, max_length):
		# word_list_copy = copy.deepcopy(word_list)
		len_lst = len(word_list)
		i=0
		while True:
			if word_list[i] == "n't":
				if i==0:
					continue
				word_list.pop(i)
				word_list[i-1] = word_list[i-1]+"n't"
				len_lst-=1
			else:
				i+=1
			if i>=min(len_lst,max_length):
				break
		return word_list

	def add_knowledge_with_vm(self, sentence, max_length, **kwargs):
		if self.debug:
			print('add_knowledge_with_vm')
		pivots = self.memory_bank.pivots
		word_list = word_tokenize(sentence)
		word_list = self.process_word_list(sentence, word_list, max_length)
		trunk = SentenceTrunk()
		trunk.push_back(BranchBundle(CLS_TOKEN))
		table_keys = self.lookup_table.keys()
		for i,w in enumerate(word_list):
			if i>max_length:
				break
			if w in pivots and w in table_keys:
				w = w.lower()
				branches = random.sample(self.lookup_table[w], 
							k=min(len(self.lookup_table[w]), self.max_facts))
				sentence_branch = BranchBundle(w, branches)
				trunk.push_back(sentence_branch)
			else:
				trunk.push_back(BranchBundle(w))
		flattened_sentence, pad_mask, trunk_len = trunk.flatten(max_length, debug=self.debug) 
		pad_mask[trunk_len] = 1
		flattened_sentence[trunk_len] = SEP_ID
		vm = trunk.get_vm(max_length, debug=self.debug)
		pos = trunk.get_pos(max_length, debug=self.debug)
		# edit vm to mask the padding
		vm *= (pad_mask.reshape(-1,1).dot(pad_mask.reshape(1,-1))!=0)
		vm = np.logical_or(vm, np.eye(max_length)).astype(np.float) # padding only attend to self

		return flattened_sentence, pos, vm, pad_mask


if __name__ == '__main__':
	sent = 'a a a a a a a a'
	words = word_tokenize(sent)
	trunk=SentenceTrunk()
	# b = BranchBundle('Hello',[Triplet(['Hi','is','Hello'])])
	# trunk.push_back(b)
	# b = BranchBundle('world')
	# trunk.push_back(b)
	# b = BranchBundle('boom',[Triplet(['boom','ba','bang'])])
	# trunk.push_back(b)
	for w in words:
		trunk.push_back(BranchBundle(w))

	flattened_sentence, pad_mask, non_pad_len = trunk.flatten(8)
	pad_mask[non_pad_len] = 1
	flattened_sentence[non_pad_len] = SEP_ID
	vm = trunk.get_vm(8)
	vm *= (pad_mask.reshape(-1,1).dot(pad_mask.reshape(1,-1))!=0)
	vm = np.logical_or(vm, np.eye(8)).astype(np.float)
	pos = trunk.get_pos(8)
	print(flattened_sentence)
	print(pad_mask)
	print(vm)
	print(pos)





