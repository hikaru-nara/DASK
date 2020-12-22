import sys
import os
import torch
import json
import random
import argparse
import collections
import time
import torch.nn as nn
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.model_builder import build_model
from uer.utils.optimizers import  BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from utils.utils import get_optimizer
from dataset import dataset_factory
from trainers import trainer_factory
from utils.readers import reader_factory
from model import model_factory
from evaluator import evaluator_factory
from optimizers import optimizer_factory
from loss import loss_factory
from utils.utils import create_logger, consistence, load_pivots
from utils.config import load_causal_hyperparam
from collate_fn import collate_factory_train,collate_factory_eval

from tensorboardX import SummaryWriter


if __name__=='__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# Path options.
	parser.add_argument("--pretrained_model_path", default=None, type=str,
						help="Path of the pretrained model.")
	parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
						help="Path of the output model.")
	parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
						help="Path of the vocabulary file.")
	# parser.add_argument("--train_path", type=str, required=True,
	#                     help="Path of the trainset.")
	# parser.add_argument("--dev_path", type=str, required=True,
	#                     help="Path of the devset.") 
	# parser.add_argument("--test_path", type=str, required=True,
						# help="Path of the testset.")
	parser.add_argument("--config_path", default="./models/google_config.json", type=str,
						help="Path of the config file.")
	parser.add_argument('--log_dir', required=True)
	parser.add_argument('--print_freq', default=100, type=int)

	# Model options.
	parser.add_argument("--batch_size", type=int, default=32,
						help="Batch size.")
	parser.add_argument('--val_batch_size', type=int, default=32)
	parser.add_argument("--seq_length", type=int, default=256,
						help="Sequence length.")
	parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
												   "cnn", "gatedcnn", "attn", \
												   "rcnn", "crnn", "gpt", "bilstm"], \
												   default="bert", help="Encoder type.")
	parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
	parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
						help="Pooling type.")
	parser.add_argument("--stages", type=str, default='6,6', help="stages for two_stage_kbert")
	parser.add_argument('--fuser', type=str, default='cross-attention', help='fuser method')
	parser.add_argument('--skip_double_embedding', action='store_true', help='whether to skip embedding at stage 2')
	parser.add_argument('--only_first_vm', action='store_true', help='only use vm in the first layer')

	# Subword options.
	parser.add_argument("--subword_type", choices=["none", "char"], default="none",
						help="Subword feature type.")
	parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
						help="Path of the subword vocabulary file.")
	parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
						help="Subencoder type.")
	parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

	# Tokenizer options.
	parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
						help="Specify the tokenizer." 
							 "Original Google BERT uses bert tokenizer on Chinese corpus."
							 "Char tokenizer segments sentences into characters."
							 "Word tokenizer supports online word segmentation based on jieba segmentor."
							 "Space tokenizer segments sentences into words according to space."
							 )

	# Optimizer options.
	parser.add_argument("--learning_rate", type=float, default=2e-5,
						help="Learning rate.")
	parser.add_argument("--warmup", type=float, default=0.1,
						help="Warm up value.")
	parser.add_argument('--momentum', default=0.9)
	parser.add_argument('--weight_decay', default=0.0001, type=float)


	# Training options.
	parser.add_argument("--dropout", type=float, default=0.5,
						help="Dropout.")
	parser.add_argument("--epochs_num", type=int, default=5,
						help="Number of epochs.")
	parser.add_argument("--report_steps", type=int, default=100,
						help="Specific steps to print prompt.")
	parser.add_argument("--seed", type=int, default=7,
						help="Random seed.")
	parser.add_argument('--supervision_rate', default=0.1, help='semi supervision rate on target domain')
	parser.add_argument('--optimizer', default='adam')
	parser.add_argument('--gpus', default='0')
	parser.add_argument('--fp16', action='store_true', help='use apex fp16 mixprecision training')
	parser.add_argument('--freeze_bert', action='store_true', help='freeze bert for fine_tune')
	parser.add_argument('--init', default='normal', type=str, help='Initialize method')
	parser.add_argument('--gamma', default=1.0, type=float)

	# Evaluation options.
	parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")
	parser.add_argument('--save_attention_mask', action="store_true", help="save attention mask of all heads from all layers on the first minibatch")

	# kg
	parser.add_argument("--kg_path", default='', help="KG path")
	parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")
	parser.add_argument('--use_kg', action='store_true', help='use knowledge graph')
	parser.add_argument('--pos_require_knowledge', type=str, help='the part of speech that \
		requires kg to add knowledge, choose a subset from [ADJ, ADP, ADV, CONJ, DET, NOUN, \
		NUM, PRT, PRON, VERB, ., X], split with "," e.g. ADJ,ADP,ADV', default='ADJ,ADV,NOUN')
	parser.add_argument('--use_pivot_kg', action='store_true')
	parser.add_argument('--num_pivots', type=int, default=2000)
	parser.add_argument('--min_occur', type=int, default=5)



	# graph-causal-DA overall options
	parser.add_argument('--task', required=True, type=str, help='[domain_adaptation/causal_inference]')
	parser.add_argument('--model_name', type=str, help='model name in model_factory')
	parser.add_argument('--dataset', type=str, help='task name in dataset_factory')
	parser.add_argument('--num_gpus', type=int, default=1, help='task name in dataset_factory')
	parser.add_argument('--num_workers', type=int, default=1, help='num worker for dataloader')
	parser.add_argument('--sparsity_lambda', type=float, default=1)
	parser.add_argument('--continuity_lambda', type=float, default=5)
	parser.add_argument('--diff_lambda', type=float, default=10, help='lambda balance term in loss')

	# DA
	parser.add_argument('--source', type=str, help='if use bdek dataset, specify with bdek.domain, e.g.\
						bdek.books')
	parser.add_argument('--target', type=str, help='if use bdek dataset, specify with bdek.domain, e.g.\
						bdek.books')
	
	# causal 
	parser.add_argument('--pollution_rate', default=[0.9, 0.7], help='pollution rate in datareader')
	parser.add_argument('--sparsity_percentage', default=0.2, help='hyperparameter in sparsity loss')
	# parser.add_argument('--repeat', required=True) deprecated



	args = parser.parse_args()

	# Load the hyperparameters from the config file.
	args = load_hyperparam(args)
	args = load_causal_hyperparam(args)
	vocab = Vocab()
	vocab.load(args.vocab_path)
	args.vocab = vocab
	args.stages = [eval(n) for n in args.stages.split(',')]
	# args.target = 'bert'
	# train_dataset = bdek_train_dataset(args.source, args.target, args.kg_path, args.supervision_rate)
	# print('train_dataset loaded')
	# eval_dataset = bdek_eval_dataset(train_dataset.evaluation_data)
	args.kg_path = args.kg_path.split(',\n\t')
	if args.use_pivot_kg:
		args.vocab_require_knowledge = load_pivots(args)
	else:
		args.vocab_require_knowledge = None
	if args.task == 'domain_adaptation':
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

		dataset = dataset_factory[args.task](args, source_reader, target_reader, graph_path=args.kg_path)
		train_dataset, dev_dataset, eval_dataset = dataset.split()

		# if '.' in args.dataset:
		# 	lst = args.dataset.split('.')
		# 	dname, domname = lst[0], lst[1]
		# 	data_reader = reader_factory[dname](domname, 'source')
		# else:
		# 	data_reader = reader_factory[args.dataset](args.pollution_rate, causal=(args.task=='causal_inference'))

		# dataset = dataset_factory[args.task](args, data_reader, graph_path=args.kg_path, vocab=args.vocab)
		# train_dataset, dev_dataset, eval_dataset = dataset.split()
	elif args.task == 'causal_inference' or args.task == 'sentim':
		if '.' in args.dataset:
			lst = args.dataset.split('.')
			dname, domname = lst[0], lst[1]
			data_reader = reader_factory[dname](domname, 'source')
		else:
			data_reader = reader_factory[args.dataset](args.pollution_rate, causal=(args.task=='causal_inference'))

		dataset = dataset_factory[args.task](args, data_reader, graph_path=args.kg_path, vocab=args.vocab)
		train_dataset, dev_dataset, eval_dataset = dataset.split()
	# if 'bdek' in source:
	#     reader = 

	# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=collate_fn_eval)
	# eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers, collate_fn=collate_fn_eval)
	collate_fn_train = collate_factory_train[args.model_name]
	collate_fn_eval = collate_factory_eval[args.model_name]
	train_sampler = torch.utils.data.RandomSampler(train_dataset)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, \
											    sampler=train_sampler, collate_fn=collate_fn_train)
	dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_eval)
	eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_eval)
	print('model init')
	
	
	# device_ids = range(args.num_gpus)
	model = model_factory[args.model_name](args)
	# print(model)
	if args.pretrained_model_path is not None:
		# Initialize with pretrained model.
		pretrained_state_dict = torch.load(args.pretrained_model_path)
		if args.model_name == 'kbert_two_stage_sentim':
			from utils.utils import load_pretrain_for_two_stage_kbert
			load_pretrain_for_two_stage_kbert(model, pretrained_state_dict)
		# print(model.state_dict.keys())
		else:
			state_dict = {}
			for k,v in pretrained_state_dict.items():
				spk = k.split('.')
				# if spk[0]=='bert':
					
				if spk[-2]=='LayerNorm':
					spk[-1] = 'weight' if spk[-1]=='gamma' else 'bias'

				k = '.'.join(spk)
				state_dict[k]=v
				
			# print(pretrained_state_dict.keys())
			# print('----------------------------')
			model.load_state_dict(state_dict, strict=False)  
	# model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if torch.cuda.device_count() > 0:
		print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
		model = nn.DataParallel(model)
	else:
		# line c=model.children.__next__() requires you use dataparallel, which is why it's ugly
		print('no parallel may cause error')

	model = model.to(device)
	
	print('optimizer init')
	# this is ugly
	c = model.children().__next__() # get the wrapped model
	optimizers = optimizer_factory[args.model_name](args, c, total_steps=len(train_dataset)*args.epochs_num//args.batch_size+1)

	# _, bert = submodule.__next__()
	# optimizers = {name:get_optimizer(args, subm.parameters(), lr[name]) for name, subm in submodule}
	# param_optimizer = list(bert.named_parameters())
	# no_decay = ['bias', 'gamma', 'beta']
	# optimizer_grouped_parameters = [
	#             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
	#             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
	# ]
	# optimizers['bert'] = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

	# optimizer = get_optimizer(args, model)

	# assert consistence(optimizers, model)
	print('optimizer num', len(optimizers))

	writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard'))
	logger = create_logger(args.log_dir)
	logger.info(args)

	criterion = loss_factory[args.model_name](args).to(device)

	total_steps = len(train_dataset)
	trainer = trainer_factory[args.model_name](args, train_loader, model, criterion, optimizers, total_steps, logger, writer=writer)
	dev_evaluator = evaluator_factory[args.model_name](args, dev_loader, model, criterion, logger)
	evaluator = evaluator_factory[args.model_name](args, eval_loader, model, criterion, logger)


	# if args.init == 'uniform':
	# 	for n, p in list(model.named_parameters()):
	# 		torch.nn.init.uniform_(p, 0.0, 1.0)
	# elif args.init == 'norm':
	# 	for n, p in list(model.named_parameters()):
	# 		torch.nn.init.normal_(p, 0.0, 1.0)


	# else:
	# 	# Initialize with normal distribution.
	# 	for n, p in list(model.named_parameters()):
	# 		if 'gamma' not in n and 'beta' not in n:
	# 			p.data.normal_(0, 0.02)


	
	global_steps = 0
	best_dev_acc = 0
	best_test_acc = 0
	for epoch in range(args.epochs_num):
		logger.info('---------------------EPOCH {}---------------------'.format(epoch))
		global_steps = trainer.train_one_epoch(device,epoch)
		dev_acc = dev_evaluator.eval_one_epoch(device)
		test_acc = evaluator.eval_one_epoch(device)
		# acc = 1.0
		if dev_acc>best_dev_acc:
			best_dev_acc = dev_acc
			best_test_acc = test_acc
			logger.info('=> saving checkpoint to {}'.format(args.log_dir))
			torch.save(model.state_dict(), os.path.join(args.log_dir, 'model_best.pth'))
		logger.info('Best dev Accuracy is {0:.4f}'.format(best_dev_acc))
		logger.info('Corresponding test Accuracy is {0:.4f}'.format(best_test_acc))

	

