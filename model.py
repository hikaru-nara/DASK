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



class Attention_Layer(nn.Module):
	def __init__(self, h_dim):
		super(Attention_Layer, self).__init__()
		self.ac = torch.nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, 2)
		)

	def forward(self, embedding):
		# return self.ac1(embedding)
		
		attention = self.ac(embedding)
		tmp = torch.max(attention,-1,keepdim=True)[0]
		attention_hard = torch.eq(attention, tmp).type_as(attention)
		tmp = (attention_hard-attention).detach()
		attention_mask = tmp + attention
		return embedding * attention_mask[:,:,0].unsqueeze(2), attention_mask
		# return embedding, attention_mask



class BertCausal(nn.Module):
	def __init__(self, args):
		super(BertCausal, self).__init__()
		config = BertConfig.from_pretrained('bert-base-uncased')
		self.bert = BertModel(config=config, add_pooling_layer=False)
		self.labels_num = 2
		self.sc = torch.nn.Sequential(
			nn.Linear(args.hidden_size, args.hidden_size),
			nn.ReLU(),
			nn.Linear(args.hidden_size, self.labels_num)
		)
		self.env_sc = torch.nn.Sequential(
			nn.Linear(args.hidden_size+1, args.hidden_size),
			nn.ReLU(),
			nn.Linear(args.hidden_size, self.labels_num)
		)
		self.causal_masking = Attention_Layer(args.hidden_size)
		# self.add_module(name='sc', module=classifier1)
		# self.add_module(name='env_sc', module=classifier2)
		# self.add_module(name='causal_masking', module=attention)

		self.use_vm = False if args.no_vm else True
		self.pooling = args.pooling
		print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

	def forward(self, src, mask, env, pos=None, vm=None):
		output = self.bert(src, mask, position_ids=pos, visible_matrix=vm)[0]

		seq_length = src.shape[1]
		env_masked = env[:,None].repeat(1,seq_length) * mask
		env_masked = env_masked[:,:,None]
		masked_output, attention_mask = self.causal_masking(output)
		assert env.dim()==1
		env_masked_output = torch.cat([masked_output, env_masked.float()],-1)
		env_logits = self.env_sc(env_masked_output)

		logits = self.sc(masked_output)
		if self.pooling == "mean":
			logits = torch.mean(logits, dim=1)
			env_logits = torch.mean(env_logits, dim=1)
		elif self.pooling == "max":
			logits = torch.max(logits, dim=1)[0]
			env_logits = torch.max(env_logits, dim=1)[0]
		elif self.pooling == "last":
			logits = logits[:, -1, :]
			env_logits = env_logits[:, -1, :]
		else:
			logits = logits[:, 0, :]
			env_logits = env_logits[:, 0, :]

		return logits, env_logits, attention_mask


class BertClassifier(nn.Module):
	def __init__(self, args):
		super(BertClassifier, self).__init__()
		config = BertConfig.from_pretrained('bert-base-uncased')
		self.bert = BertModel(config=config, add_pooling_layer=False)
		self.labels_num = 2
		self.classifier = torch.nn.Sequential(
			nn.Linear(args.hidden_size, args.hidden_size),
			nn.Dropout(args.dropout),
			nn.ReLU(),
			nn.Linear(args.hidden_size, self.labels_num)
		)
		
		self.use_vm = False if args.no_vm else True
		self.pooling = args.pooling
		print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

	def forward(self, src, mask, pos=None, vm=None, output_attention=False):
		"""
		Args:
			src: [batch_size x seq_length]
			label: [batch_size]
			mask: [batch_size x seq_length]
		"""
		if output_attention:
			outputs = self.bert(src, mask, position_ids=pos, visible_matrix=vm, output_attentions=output_attention)
			output, _, all_attentions = outputs[0], outputs[1], outputs[2]
		else:
			output = self.bert(src, mask, position_ids=pos, visible_matrix=vm, output_attentions=output_attention)[0]
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
		
		if output_attention:
			
			return self.classifier(output), list(all_attentions)
		else:
			return self.classifier(output)


class linearfuser(nn.Module):
	def __init__(self, args):
		super(linearfuser, self).__init__()
		self.fuser = torch.nn.Sequential(
				nn.Linear(args.hidden_size*2, args.hidden_size),
				nn.Dropout(args.dropout),
				nn.ReLU(),
				nn.Linear(args.hidden_size, args.hidden_size)
			)

	def forward(self, output1, output2):
		output = torch.cat([output1, output2], dim=2)
		return self.fuser(output)


class KBert_two_branch(nn.Module):
	def __init__(self, args):
		super(KBert_two_branch, self).__init__()
		config = BertConfig.from_pretrained('bert-base-uncased')
		bert_list = nn.ModuleList()
		for num_layer in args.stages:
			config.num_hidden_layers = num_layer
			bert_list.append(BertModel(config=config, add_pooling_layer=False))
		self.bert = bert_list
		# self.bert = BertModel(config=config, add_pooling_layer=False)
		self.labels_num = 2
		self.classifier = torch.nn.Sequential(
			nn.Linear(args.hidden_size, args.hidden_size),
			nn.Dropout(args.dropout),
			nn.ReLU(),
			nn.Linear(args.hidden_size, self.labels_num)
		)
		fuser = args.fuser
		if fuser == 'linear':
			self.fuser = linearfuser(args)
		elif fuser == 'cross-attention':
			from pytorch_transformers.kbert_model import BertCrossAttention
			self.fuser = BertCrossAttention(config)
		else:
			raise NotImplementedError

		self.use_vm = False if args.no_vm else True
		self.pooling = args.pooling

	def forward(self, tokens_kg, tokens_org, mask_kg, mask_org, pos, vm, output_attention=False):
		"""
		Args:
			tokens_kg: [batch_size x seq_length], sentence with knowledge from graph
			tokens_org: [batch_size x seq_length], sentence without extra knowledge, but contain more part of orginal sentence
			mask_kg: padding mask for tokens_kg
			mask_org: padding mask for tokens_org
			pos: position_id for tokens_kg
			vm: visible_matrix for tokens_kg
			output_attention: whether or not to output attention mask for each layer and each attention head
		"""
		output_kg = self.bert[0](tokens_kg, mask_kg, position_ids=pos, visible_matrix=vm, output_attentions=output_attention)[0]
		output_org = self.bert[0](tokens_org, mask_org, output_attentions=output_attention)[0]
		# B, N, 2C
		# output_stage1 = torch.cat([output_kg, output_org], dim=2)
		input_stage2 = self.fuser(output_kg, output_org)
		output_stage2 = self.bert[1](inputs_embeds=input_stage2)[0] # one extra embedding layer here

		if self.pooling == "mean":
			output = torch.mean(output_stage2, dim=1)
		elif self.pooling == "max":
			output = torch.max(output_stage2, dim=1)[0]
		elif self.pooling == "last":
			output = output_stage2[:, -1, :]
		else:
			output = output_stage2[:, 0, :]
		return self.classifier(output)
		
		# if output_attention:
			
		# 	return self.classifier(output), list(all_attentions)
		# else:
		# 	return self.classifier(output)


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
				'causal': BertCausal,
				'kbert_two_stage_sentim': KBert_two_branch,
				'sentim': BertClassifier,
				'base_DA': BertClassifier
				}



