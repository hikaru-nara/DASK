# -*- encoding:utf-8 -*-
from __future__ import unicode_literals

# bert Special token ids.
PAD_ID = 0
UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103

# bert Special token words.
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'


NEVER_SPLIT_TAG = [
    PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN
]
def initialize_constant(model_name):
	print(model_name)
	if 'roberta' in model_name:
		print('is roberta')
		# Special token ids.
		global PAD_ID, UNK_ID, SEP_ID, MASK_ID, PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN
		PAD_ID = 1
		UNK_ID = 3
		CLS_ID = 0
		SEP_ID = 2
		MASK_ID = 50264

		# Special token words.
		PAD_TOKEN = '<pad>'
		UNK_TOKEN = '<unk>'
		CLS_TOKEN = '<s>'
		SEP_TOKEN = '</s>'
		MASK_TOKEN = '<mask>'