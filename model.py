import torch
import torch.nn as nn
import json
import random
# import argparse

from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.model_builder import build_model
from uer.utils.optimizers import  BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from torch.autograd import Function
# from transformers import BertModel, BertConfig, BertTokenizer
from pytorch_transformers.kbert_model import BertModel
from pytorch_transformers.model_config import BertConfig

class ReverseLayerF(Function):
	'''
	copy from KinGDOM
	'''
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha

		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha

		return output, None

class BertSentimClassifier(nn.Module):
	def __init__(self, args):
		super(BertSentimClassifier, self).__init__()
		self.pooling = args.pooling
		self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
		self.output_layer_2 = nn.Linear(args.hidden_size, 2)

	def forward(self, output):
		if self.pooling == "mean":
			output = torch.mean(output, dim=1)
		elif self.pooling == "max":
			output = torch.max(output, dim=1)[0]
		elif self.pooling == "last":
			output = output[:, -1, :]
		else:
			output = output[:, 0, :]
		output = torch.tanh(self.output_layer_1(output))
		logits = self.output_layer_2(output)
		return logits


class BertClassifier(nn.Module):
	def __init__(self, args):
		super(BertClassifier, self).__init__()
		config = BertConfig.from_pretrained('bert-base-uncased')
		self.bert = BertModel(config=config, add_pooling_layer=False)
		# self.add_module(name='bert', module=model)
		# self.embedding = model.embedding
		# self.encoder = model.encoder
		self.labels_num = 2
		classifier = torch.nn.Sequential(
			nn.Linear(args.hidden_size, args.hidden_size),
			nn.ReLU(),
			nn.Linear(args.hidden_size, self.labels_num)
		)
		self.add_module(name='sc', module=classifier)
		# self.pooling = 'max'
		# self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
		# self.output_layer_2 = nn.Linear(args.hidden_size, 2)
		# self.softmax = nn.LogSoftmax(dim=-1)
		# self.criterion = nn.NLLLoss()
		self.use_vm = False if args.no_vm else True
		self.pooling = args.pooling
		print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

	def forward(self, src, mask, pos=None, vm=None):
		"""
		Args:
			src: [batch_size x seq_length]
			label: [batch_size]
			mask: [batch_size x seq_length]
		"""
		# Embedding.
		# assert pos is None
		# assert vm is None
		# modules = {name: module for name, module in self.named_children()}
		# embedding = modules['bert'].embeddings(src, mask, pos)
		# if not self.use_vm:
		# 	vm = None
		# output = modules['bert'].encoder(embedding, mask, vm)
		# # emb = self.embedding(src, mask, pos)
		# # Encoder.
		
		# # output = self.encoder(emb, mask, vm)
		# # Target.
		# logits = modules['sc'](output)
		# # loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))

		output = self.bert(src, mask, position_ids=pos, visible_matrix=vm)[0]
		# print(output.shape)
		# Target.
		if self.pooling == "mean":
			output = torch.mean(output, dim=1)
		elif self.pooling == "max":
			output = torch.max(output, dim=1)[0]
		elif self.pooling == "last":
			output = output[:, -1, :]
		else:
			output = output[:, 0, :]
		modules = {name: module for name, module in self.named_children()}
		classifier = modules['sc']

		return classifier(output)


class BertForDA(nn.Module):
	def __init__(self, args):
		super(BertForDA, self).__init__()
		print("Initializing main bert model...")
		model_name = 'bert-base-uncased'
		model_config = BertConfig.from_pretrained(model_name)
		self.bert_model = BertModel.from_pretrained(model_name, config=model_config)
		self.labels_num = 2
		self.pooling = args.pooling
		classifier = torch.nn.Sequential(
			nn.Linear(args.hidden_size, args.hidden_size),
			nn.ReLU(),
			nn.Linear(args.hidden_size, self.labels_num)
		)
		self.add_module(name='classifier', module=classifier)
		pooler = torch.nn.Sequential(
			nn.Linear(args.hidden_size, args.hidden_size),
			nn.ReLU(),
			nn.Linear(args.hidden_size, args.hidden_size)
		)
		self.add_module(name='pooler', module=pooler)
		self.softmax = nn.LogSoftmax(dim=-1)
		self.criterion = nn.NLLLoss()

	def forward(self, input_ids, masks, pos=None, vm=None, visualize=False):
		"""
		Args:
			input_ids: [batch_size x seq_length]
			labels: [batch_size]
			masks: [batch_size x seq_length]
		"""
		output = self.bert_model(input_ids, masks)[0]
		print(output.shape)
		# Target.
		if self.pooling == "mean":
			output = torch.mean(output, dim=1)
		elif self.pooling == "max":
			output = torch.max(output, dim=1)[0]
		elif self.pooling == "last":
			output = output[:, -1, :]
		else:
			output = output[:, 0, :]
		modules = {name: module for name, module in self.named_children()}
		classifier = modules['classifier']
		pooler = modules['pooler']
		# loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
		return classifier(output)


class Sentiment_Classifier(nn.Module):
	def __init__(self, h_dim, max_sentence_length, drop):
		super(Sentiment_Classifier, self).__init__()
		self.sc1 = nn.Linear(h_dim, 100)
		self.sc2 = nn.MaxPool1d(kernel_size=max_sentence_length)
		self.sc3 = nn.Linear(100, 10)
		self.sc4 = nn.Linear(10, 2)
		self.sc_activate = nn.Tanh()
		self.sc_drop = nn.Dropout(drop)

	def forward(self, embedding):
		# print('------------------sentim cls--------------------')
		h = self.sc1(embedding)
		# print(h)
		# print([p1 for p1 in self.sc1.parameters()])
		p = self.sc2(h.permute(0,2,1)).squeeze()
		# print(p)
		
		h = self.sc3(p)
		# print(h)
		# print([p1 for p1 in self.sc3.parameters()])
		p = self.sc_drop(h)
		# print(p)
		h = self.sc_activate(p)
		# print(h)
		p = self.sc4(h)
		# print(p)
		# print([p1 for p1 in self.sc4.parameters()])
		# h = self.sc_softmax(p)
		# print(h)

		return p


class Attention_Layer(nn.Module):
	def __init__(self, h_dim):
		super(Attention_Layer, self).__init__()
		self.ac1 = nn.Linear(h_dim,10)
		self.ac2 = nn.Linear(10,2)

	def forward(self, embedding):
		# return self.ac1(embedding)
		feature = self.ac1(embedding)
		attention = self.ac2(feature)
		tmp = torch.max(attention,-1,keepdim=True).values
		attention_hard = torch.eq(attention, tmp).type_as(attention)
		tmp = (attention_hard-attention).detach()
		attention_mask = tmp + attention
		return embedding * attention_mask[:,:,0].unsqueeze(2), attention_mask
		# return embedding, attention_mask


class causal_inference_net(nn.Module):
	def __init__(self, args):
		super(causal_inference_net, self).__init__()
		kbert_model = build_model(args)
		self.add_module(name='bert', module=kbert_model)
		

		self.max_sentence_length = args.seq_length
		self.h_dim = args.hidden_size
		self.drop = args.dropout
		# attention layer
		al = Attention_Layer(self.h_dim)
		self.add_module(name='al', module=al)
		# self.extractor = nn.ModuleDict({
		# 		'kbert': kbert_model,
		# 		'al': al
		# 	})

		# sentiment classifier
		sc = Sentiment_Classifier(self.h_dim, self.max_sentence_length, self.drop)
		self.add_module(name='sc', module=sc)
		# environment enable classifier 
		# 1 is the env feature dim, should be altered 
		env_sc = Sentiment_Classifier(self.h_dim+1, self.max_sentence_length, self.drop)
		self.add_module(name='env_sc', module=env_sc)
		
		
		# self.ac1 = nn.Linear(self.h_dim, 10)
		# self.ac2 = nn.Linear(10, 2)

	def param_groups(self):
		return [self.sentiment_params(), self.env_enable_params(), self.extractor_params()]

	def sentiment_params(self):
		return self.sc.parameters()

	def env_enable_params(self):
		return self.ec.parameters()

	def extractor_params(self):
		return self.kbert_model.parameters() + self.ac1.parameters() + self.ac2.parameters()

	def forward(self, tokens, mask, pos, vm, aug):
		modules = {name: module for name, module in self.named_children()}
		embedding = modules['bert'].embedding(tokens, mask, pos)
		output = modules['bert'].encoder(embedding, mask, vm)

		rationale_feature, attention_mask = modules['al'](output)
		# print(aug)÷

		aug_ = aug.unsqueeze(1).unsqueeze(2).repeat(1,mask.shape[1],1).type_as(rationale_feature)

		# assert aug_.dim() == 3
		augmented_feature = torch.cat([aug_, rationale_feature], dim=2)
		sentiment_probs = modules['sc'](rationale_feature)
		env_enable_probs = modules['env_sc'](augmented_feature)

		return sentiment_probs, env_enable_probs, attention_mask




class graph_domain_adaptation_net(nn.Module):
	# incomplete
	def __init__(self, args):
		super(graph_domain_adaptation_net, self).__init__()
		self.kbert_model = build_model(args)
		self.max_sentence_length = args.seq_length
		self.h_dim = args.hidden_dim
		self.drop = args.dropout
		print('graph_domain_adaptation_net')
		print(max_sentence_length)
		print(hidden_dim)


		self.sc1 = nn.Linear(hidden_dim, 100)
		self.sc2 = nn.MaxPool1d(kernel_size=max_sentence_length)
		self.sc3 = nn.Linear(100, 10)
		self.sc4 = nn.Linear(10, 2)
		self.sc_relu = nn.ReLU()
		self.sc_drop = nn.Dropout(self.drop)
		self.sc_softmax = nn.Softmax(dim=-1)
		self.sc = nn.Sequential(self.sc1, self.sc2, self.sc3, self.sc4, self.sc_relu, self.sc_drop, self.sc_softmax)

		self.dc1 = nn.Linear(hidden_dim, 100)
		self.dc2 = nn.MaxPool1d(kernel_size=max_sentence_length)
		self.dc3 = nn.Linear(100, 10)
		self.dc4 = nn.Linear(10, 2)
		self.dc_relu = nn.ReLU()
		self.dc_drop = nn.Dropout(self.drop)
		self.dc_softmax = nn.Softmax(dim=-1)
		self.dc = nn.Sequential(self.dc1, self.dc2, self.dc3, self.dc4, self.dc_relu, self.dc_drop, self.dc_softmax)

		self.ac1 = nn.Linear(hidden_dim, 10)
		self.ac2 = nn.Linear(10, 2)

	def param_groups(self):
		return [self.sentiment_params(), self.domain_params(), self.extractor_params()]

	def sentiment_params(self):
		return self.sc.parameters()

	def domain_params(self):
		return self.dc.parameters()

	def extractor_params(self):
		return self.kbert_model.parameters() + self.ac1.parameters() + self.ac2.parameters()

	def sentiment_classifier(self, embedding):
		'''
		embedding shape: [batch_size, sentence_max_length, h_dim]
		'''
		length = embedding.shape[1]
		h = self.sc1(embedding)
		p = self.sc2(h.permute(0,2,1))
		h = self.sc3(p)
		p = self.sc_drop(h)
		h = self.sc_relu(p)
		p = self.sc4(h)
		h = self.sc_softmax(p)

		return h

	def domain_classifier(self, embedding):
		'''
		embedding shape: [batch_size, sentence_max_length, h_dim]
		'''
		length = embedding.shape[1]
		h = self.dc1(embedding)
		p = self.dc2(h.permute(0,2,1))
		h = self.dc3(p)
		p = self.dc_drop(h)
		h = self.dc_relu(p)
		p = self.dc4(h)
		h = self.dc_softmax(p)

		return h

	def forward(self, tokens, mask, pos, vm):
		embedding = self.kbert_model.embedding(tokens, mask, pos)
		output = self.kbert_model.encoder(embedding, mask, vm)
		
		R_feature = ReverseLayerF.apply(feature)
		sentiment_output = self.sentiment_classifier(feature)
		domain_output = self.domain_classifier(R_feature)
		return sentiment_output, domain_output

model_factory = {
				'graph_causal': None,
				'graph': graph_domain_adaptation_net,
				'causal': causal_inference_net,
				# 'sentim': BertClassifier
				'sentim': BertClassifier
				}




