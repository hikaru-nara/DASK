import torch 
from utils.utils import AverageMeter
from utils.utils import accuracy, consistence, get_optimizer
import time
from tqdm import tqdm
import numpy as np 


class Base_sentim_Trainer(object):
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, logger, writer=None, tokenizer=None):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0
		self.tokenizer = tokenizer

	def train_one_epoch(self, device, epoch=0):
		loss_meter = AverageMeter()
		acc_meter = AverageMeter()
		time_meter = AverageMeter()
		self.model.train()
		for i, labeled_data in enumerate(self.train_loader):
			self.optimizers.scheduler_step()
			# print(optimizers.optimizers['bert'].get_lr()[0])
			self.model.zero_grad()
			input_ids, masks, labels = \
				(labeled_data[k].to(device) for k in ['tokens', 'mask', 'label'])
			labels = labels.long()
			# input_ids_s, masks_s, aug_input_ids_s, aug_masks_s, doms_s = \
			# 	(unlabeled_src_data[k].to(device) for k in ['tokens', 'mask', 'aug_tokens', 'aug_mask', 'domain'])
			# input_ids_t, masks_t, aug_input_ids_t, aug_masks_t, doms_t = \
			# 	(unlabeled_tgt_data[k].to(device) for k in ['tokens', 'mask', 'aug_tokens', 'aug_mask', 'domain'])
			# input_ids, masks, labels = input_ids.long().to(device), masks.long().to(device), labels.long().to(device)
			start_time = time.time()
			logits = self.model(input_ids, masks)
			loss = self.loss_criterion(logits, labels)
			acc = accuracy(logits.detach().cpu().numpy(), labels.detach().cpu().numpy())
			self.optimizers.step(loss)
			end_time = time.time()
			time_meter.update(end_time - start_time)
			loss_meter.update(float(loss))
			acc_meter.update(float(acc))

			if i % self.args.print_freq == 0:
				log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter,
						sentim_loss=loss_meter,
						sentim_acc=acc_meter)
				self.logger.info(log_string)
			self.global_steps += 1

		return self.global_steps


class kbert_two_stage_sentim_Trainer(object):
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, logger):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0

	def train_one_epoch(self, device):
		args = self.args
		train_loader = self.train_loader
		model = self.model
		loss_criterion = self.loss_criterion
		optimizers = self.optimizers
		total_steps = self.total_steps
		logger = self.logger

		sentim_loss_meter = AverageMeter()
		sentim_acc_meter = AverageMeter()
		time_meter = AverageMeter()
		model.train()
		
		for i,batch_data in enumerate(tqdm(train_loader)):
			self.optimizers.scheduler_step()
			# print(optimizers.optimizers['bert'].get_lr()[0])
			model.zero_grad()
			tokens_kg, mask_kg, labels, tokens_org, mask_org = batch_data['tokens_kg'], batch_data['mask_kg'], \
				 batch_data['label'], batch_data['tokens_org'], batch_data['mask_org']

			if self.args.use_kg:
				positions, vms = batch_data['pos'], batch_data['vm']
			else:
				positions, vms = None, None
			labels = labels.long().to(device)
			mask_kg = mask_kg.long().to(device)
			# positions = positions.long().to(device)
			# vms = vms.long().to(device)
			tokens_org, mask_org = tokens_org.long().to(device), mask_org.long().to(device)
			tokens_kg = tokens_kg.long().to(device)

			start_time = time.time()

			sentim_probs = model(tokens_kg, tokens_org, mask_kg, mask_org, positions, vms)
			sentim_loss = loss_criterion(sentim_probs, labels)

			sentim_acc = accuracy(sentim_probs.detach().cpu().numpy(), labels.detach().cpu().numpy())

			optimizers.step(sentim_loss)

			end_time = time.time()
			time_meter.update(end_time-start_time)
			sentim_loss_meter.update(float(sentim_loss))
			sentim_acc_meter.update(float(sentim_acc))

			if i % self.args.print_freq==0:
				log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, 
						sentim_loss=sentim_loss_meter, 
						sentim_acc=sentim_acc_meter)
				logger.info(log_string)
			self.global_steps += 1

		logger.info('-----Training epoch summary------')
		log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, 
						sentim_loss=sentim_loss_meter, 
						sentim_acc=sentim_acc_meter)
		logger.info(log_string)

		return self.global_steps


class kbert_two_stage_da_Trainer(object):
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, logger):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0

	def train_one_epoch(self, device):
		args = self.args
		train_loader = self.train_loader
		model = self.model
		loss_criterion = self.loss_criterion
		optimizers = self.optimizers
		total_steps = self.total_steps
		logger = self.logger

		sentim_loss_meter = AverageMeter()
		sentim_acc_meter = AverageMeter()
		time_meter = AverageMeter()
		model.train()
		
		for i, (labeled_data, _) in enumerate(tqdm(train_loader)):
			self.optimizers.scheduler_step()
			# print(optimizers.optimizers['bert'].get_lr()[0])
			model.zero_grad()
			tokens_kg, mask_kg, labels, tokens_org, mask_org = labeled_data['tokens_kg'], labeled_data['mask_kg'], \
				 labeled_data['label'], labeled_data['tokens_org'], labeled_data['mask_org']

			if self.args.use_kg:
				positions, vms = labeled_data['pos'], labeled_data['vm']
			else:
				positions, vms = None, None
			labels = labels.long().to(device)
			mask_kg = mask_kg.long().to(device)
			# positions = positions.long().to(device)
			# vms = vms.long().to(device)
			tokens_org, mask_org = tokens_org.long().to(device), mask_org.long().to(device)
			tokens_kg = tokens_kg.long().to(device)

			start_time = time.time()

			sentim_probs = model(tokens_kg, tokens_org, mask_kg, mask_org, positions, vms)
			sentim_loss = loss_criterion(sentim_probs, labels)

			sentim_acc = accuracy(sentim_probs.detach().cpu().numpy(), labels.detach().cpu().numpy())

			optimizers.step(sentim_loss)

			end_time = time.time()
			time_meter.update(end_time-start_time)
			sentim_loss_meter.update(float(sentim_loss))
			sentim_acc_meter.update(float(sentim_acc))

			if i % self.args.print_freq==0:
				log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, 
						sentim_loss=sentim_loss_meter, 
						sentim_acc=sentim_acc_meter)
				logger.info(log_string)
			self.global_steps += 1

		logger.info('-----Training epoch summary------')
		log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, 
						sentim_loss=sentim_loss_meter, 
						sentim_acc=sentim_acc_meter)
		logger.info(log_string)

		return self.global_steps




class sentim_Trainer(object):
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, logger):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0

	def train_one_epoch(self, device):
		args = self.args
		train_loader = self.train_loader
		model = self.model
		loss_criterion = self.loss_criterion
		optimizers = self.optimizers
		total_steps = self.total_steps
		logger = self.logger

		sentim_loss_meter = AverageMeter()
		sentim_acc_meter = AverageMeter()
		time_meter = AverageMeter()
		model.train()
		
		for i,batch_data in enumerate(tqdm(train_loader)):
			self.optimizers.scheduler_step()
			# print(optimizers.optimizers['bert'].get_lr()[0])
			model.zero_grad()
			tokens, masks, labels = batch_data['tokens'], batch_data['mask'], \
				 batch_data['label']
			if self.args.use_kg:
				positions, vms = batch_data['pos'], batch_data['vm']
			else:
				positions, vms = None, None
			labels = labels.long().to(device)
			masks = masks.long().to(device)
			# positions = positions.long().to(device)
			# vms = vms.long().to(device)
			tokens = tokens.long().to(device)
			start_time = time.time()

			sentim_probs = model(tokens, masks, pos=positions, vm=vms)
			sentim_loss = loss_criterion(sentim_probs, labels)

			sentim_acc = accuracy(sentim_probs.detach().cpu().numpy(), labels.detach().cpu().numpy())

			optimizers.step(sentim_loss)

			end_time = time.time()
			time_meter.update(end_time-start_time)
			sentim_loss_meter.update(float(sentim_loss))
			sentim_acc_meter.update(float(sentim_acc))

			if i % self.args.print_freq==0:
				log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, 
						sentim_loss=sentim_loss_meter, 
						sentim_acc=sentim_acc_meter)
				logger.info(log_string)
			self.global_steps += 1

		logger.info('-----Training epoch summary------')
		log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, 
						sentim_loss=sentim_loss_meter, 
						sentim_acc=sentim_acc_meter)
		logger.info(log_string)

		return self.global_steps


class causal_Trainer(object):
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, logger, writer=None):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0
		self.writer = writer

	def train_one_epoch(self, device, epoch=0):
		args = self.args
		train_loader = self.train_loader
		model = self.model
		loss_criterion = self.loss_criterion
		optimizers = self.optimizers
		total_steps = self.total_steps
		logger = self.logger

		sentim_loss_meter = AverageMeter()
		sentim_acc_meter = AverageMeter()
		time_meter = AverageMeter()
		env_enable_sentim_loss_meter = AverageMeter()
		env_enable_sentim_acc_meter = AverageMeter()

		model.train()
		for i, batch_data in enumerate(tqdm(train_loader)):
			optimizers.scheduler_step()
			model.zero_grad()
			tokens, masks, envs, labels = batch_data['tokens'], batch_data['mask'], \
				 batch_data['env'], batch_data['label']
			if self.args.use_kg:
				positions, vms = batch_data['pos'], batch_data['vm']
				positions = positions.long().to(device)
				vms = vms.long().to(device)
			else:
				positions, vms = None, None

			labels = labels.long().to(device)
			masks = masks.long().to(device)
			envs = envs.to(device)
			tokens = tokens.long().to(device)
			start_time = time.time()

			sentim_probs, env_enable_sentim_probs, rationale_mask = model(tokens, masks, envs, positions, vms)
			# print(sentim_probs.shape)
			extractor_loss, sentim_loss, env_enable_sentim_loss = loss_criterion(sentim_probs, \
					env_enable_sentim_probs, labels, rationale_mask, masks)

			sentim_acc = accuracy(sentim_probs.detach().cpu().numpy(), labels.detach().cpu().numpy())
			env_enable_sentim_acc = accuracy(env_enable_sentim_probs.detach().cpu().numpy(), labels.detach().cpu().numpy())

			optimizers.step([extractor_loss, sentim_loss, env_enable_sentim_loss],i)

			end_time = time.time()
			time_meter.update(end_time-start_time)
			sentim_loss_meter.update(float(sentim_loss))
			sentim_acc_meter.update(float(sentim_acc))
			env_enable_sentim_acc_meter.update(float(env_enable_sentim_acc))
			env_enable_sentim_loss_meter.update(float(env_enable_sentim_loss))

			if self.writer is not None:
				writer = self.writer
				writer.add_scalar('train_sentim_loss', sentim_loss_meter.val, self.global_steps)
				writer.add_scalar('train_sentim_acc', sentim_acc_meter.val, self.global_steps)
				writer.add_scalar('train_env_sentim_loss', env_enable_sentim_loss_meter.val, self.global_steps)
				writer.add_scalar('train_env_sentim_acc', env_enable_sentim_acc_meter.val, self.global_steps)
			if i % self.args.print_freq==0:
				log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'env_enable_sentiment_loss: {env_enable_sentim_loss.val:.3f}({env_enable_sentim_loss.avg:.3f})' \
					'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})' \
					'env_enable_sentiment_acc: {env_enable_sentim_acc.val:.3f}({env_enable_sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, env_enable_sentim_loss=env_enable_sentim_loss_meter,
						sentim_loss=sentim_loss_meter, env_enable_sentim_acc=env_enable_sentim_acc_meter,
						sentim_acc=sentim_acc_meter)

				logger.info(log_string)
			self.global_steps += 1


		logger.info('-----Training epoch summary------')
		log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, 
						sentim_loss=sentim_loss_meter, 
						sentim_acc=sentim_acc_meter)
		logger.info(log_string)
		return self.global_steps




class graph_causal_Trainer(object):
	# not complete
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, logger):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0

	def train_one_epoch(self):
		'''
		main train procedure
		'''
		args = self.args
		train_loader = self.train_loader
		model = self.model
		loss_criterion = self.loss_criterion
		optimizers = self.optimizers
		total_steps = self.total_steps
		logger = self.logger

		dom_loss_meter = AverageMeter()
		sentim_loss_meter = AverageMeter()
		dom_loss_u_meter = AverageMeter()
		dom_acc_meter = AverageMeter()
		dom_acc_u_meter = AverageMeter()
		sentim_acc_meter = AverageMeter()
		time_meter = AverageMeter()
		ext_opt = optimizers[2]
		dom_opt = optimizers[1]
		sentim_opt = optimizers[0]
		model.train()
		# lamda = 2/(1+math.exp(-10*global_steps/total_steps))-1

		for i, labeled_batch, unlabeled_batch in enumerate(train_loader):
			self.global_steps += 1 
			lamda = 2/(1+math.exp(-10*global_steps/total_steps))-1
			path = global_steps % 7
			tokens, masks, positions, vms, labels, domains, _ = labeled_batch
			start_time = time.time()

			# outputs = 
			sentim, env_sentim, dom = model(tokens, masks, positions, vms)
			ext_loss, dom_loss, sentim_loss, env_enable_loss = loss_criterion(sentim, labels)
			sentim_env_loss = loss_criterion(sentim)
			dom_loss = loss_criterion(dom, domains)

			sentim_acc = accuracy(sentim, labels)
			dom_acc = accuracy(dom, domains)

			# optimizer_wrapper.step(global_steps, )
			if path in [0,1,2,3,4]:
				ext_opt.zero_grad()
				dom_loss_u.backward()
				ext_opt.step()
			elif path in [5,6]:
				dom_opt.zero_grad()
				dom_loss_u.backward()
				dom_opt.step()
			

			tokens_u, masks_u, positions_u, vms_u, domains_u, _ = unlabeled_batch
			sentim_u, dom_u = model(tokens_u, masks_u, positions_u, vms_u)
			dom_loss_u = loss_criterion(dom_u, domains_u)
			dom_acc_u = accuracy(dom_u, domains_u)

			dom_loss_meter.update(dom_loss)
			dom_loss_u_meter.update(dom_loss_u)
			sentim_loss_meter.update(sentim_loss)
			dom_acc_meter.update(dom_acc)
			dom_acc_u_meter.update(dom_acc_u)
			sentim_acc_meter.update(sentim_acc)
		
			if path in [0,1,2,3,4]:
				ext_opt.zero_grad()
				dom_loss_u.backward()
				ext_opt.step()
			elif path in [5,6]:
				dom_opt.zero_grad()
				dom_loss_u.backward()
				dom_opt.step()
			end_time = time.time()
			time_meter.update(end_time - start_time)

			if i % args.print_freq==0:
				log_string = 'Iteration[{0}]\t' \
					'time {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'domain_loss {dom_loss.val:.3f}({dom_loss.avg:.3f})\t' \
					'unlabeled_domain_loss {dom_loss_u.val:.3f}({dom_loss_u.avg:.3f})\t' \
					'sentiment_loss {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'domain_accuracy {dom_acc.val:.3f}({dom_acc.avg:.3f})\t' \
					'unlabeled_domain_accuracy {dom_acc_u.val:.3f}({dom_acc_u.avg:.3f})\t' \
					'sentiment_accuracy {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, dom_loss=dom_loss_meter, dom_loss_u=dom_loss_u_meter,
						sentim_loss=sentim_loss_meter, dom_acc=dom_acc_meter, dom_acc_u=dom_acc_u_meter,
						sentim_acc=sentim_acc_meter)

				logger.info(log_string)


		return self.global_steps



class base_DA_Trainer(object):
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, logger):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0

	def train_one_epoch(self, device, epoch=0):
		args = self.args
		train_loader = self.train_loader
		model = self.model
		loss_criterion = self.loss_criterion
		optimizers = self.optimizers
		total_steps = self.total_steps
		logger = self.logger

		sentim_loss_meter = AverageMeter()
		sentim_acc_meter = AverageMeter()
		time_meter = AverageMeter()
		model.train()
		
		for i, (labeled_batch, unlabeled_batch) in enumerate(tqdm(train_loader)):
			self.optimizers.scheduler_step()
			# print(optimizers.optimizers['bert'].get_lr()[0])
			model.zero_grad()
			tokens, masks, labels = labeled_batch['tokens'].long().to(device), labeled_batch['mask'].long().to(device), \
				 labeled_batch['label'].long().to(device)
			# tokens_u, masks_u, labels_u = unlabeled_batch['tokens'].long().to(device), unlabeled_batch['mask'].long().to(device), \
			# 	 unlabeled_batch['label'].long().to(device)
			if self.args.use_kg:
				positions, vms = labeled_batch['pos'].long().to(device), labeled_batch['vm'].long().to(device)
				# positions_u, vms_u = unlabeled_batch['pos'].long().to(device), unlabeled_batch['vm'].long().to(device)
			else:
				positions, vms, positions_u, vms_u = None, None, None, None
			

			start_time = time.time()

			sentim_probs = model(tokens, masks, pos=positions, vm=vms, only_first_vm=self.args.only_first_vm)
			sentim_loss = loss_criterion(sentim_probs, labels)

			sentim_acc = accuracy(sentim_probs.detach().cpu().numpy(), labels.detach().cpu().numpy())

			if i % self.args.balanced_interval == 0:
				optimizers.step(sentim_loss)
			else:
				# optimizers.step(sentim_loss)
				pass

			end_time = time.time()
			time_meter.update(end_time-start_time)
			sentim_loss_meter.update(float(sentim_loss))
			sentim_acc_meter.update(float(sentim_acc))

			if i % self.args.print_freq==0:
				log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, 
						sentim_loss=sentim_loss_meter, 
						sentim_acc=sentim_acc_meter)
				logger.info(log_string)
			self.global_steps += 1

		logger.info('-----Training epoch summary------')
		log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, 
						sentim_loss=sentim_loss_meter, 
						sentim_acc=sentim_acc_meter)
		logger.info(log_string)

		return self.global_steps



class DANN_Trainer(object):
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, logger, writer=None, tokenizer=None):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0
		self.tokenizer = tokenizer

	def train_one_epoch(self, device, epoch):
		start_steps = epoch * len(self.train_loader)
		total_steps = self.args.epochs_num * len(self.train_loader)
		loss_meter = AverageMeter()
		acc_meter = AverageMeter()
		dom_acc_meter = AverageMeter()
		time_meter = AverageMeter()
		self.model.train()
		for i, (labeled_data, unlabeled_data) in enumerate(tqdm(self.train_loader)):
			self.optimizers.scheduler_step()
			# print(optimizers.optimizers['bert'].get_lr()[0])
			self.model.zero_grad()
			input_ids, masks, labels, doms = \
				(labeled_data[k].to(device).long() for k in ['tokens', 'mask', 'label', 'domain'])
			input_ids_u, masks_u, doms_u = \
				(unlabeled_data[k].to(device).long() for k in ['tokens', 'mask', 'domain'])
			if self.args.use_kg:
				pos, vm = labeled_data['pos'].to(device).long(), labeled_data['vm'].to(device).long()
				pos_u, vm_u = unlabeled_data['pos'].to(device).long(), unlabeled_data['vm'].to(device).long()
			# input_ids_t, masks_t, doms_t = \
			# 	(unlabeled_tgt_data[k].to(device) for k in ['tokens', 'mask', 'domain'])

			# input_ids2 = torch.cat([input_ids_s, input_ids_t], dim=0)
			# masks2 = torch.cat([masks_s, masks_t], dim=0)
			# dom_labels2 = torch.cat([doms_s, doms_t], dim=0)
			# input_ids2 = input_ids_t
			# masks2 = masks_t
			# dom_labels2 = doms_t

			start_time = time.time()
			# setup hyperparameters
			p = float(i + start_steps) / total_steps
			constant = 2. / (1. + np.exp(-self.args.gamma * p)) - 1
			# constant = 0.5

			# forward
			class_preds, labeled_preds, unlabeled_preds = self.model(input_ids, masks, pos, vm, \
						input_ids_u, masks_u, pos_u, vm_u, constant)
			all_preds = torch.cat([labeled_preds, unlabeled_preds], dim=0)
			all_dom_labels = torch.cat([doms, doms_u], dim=0)
			loss, cls_loss, domain_loss = self.loss_criterion(class_preds, labels, all_preds, all_dom_labels)
			acc = accuracy(class_preds.detach().cpu().numpy(), labels.detach().cpu().numpy())
			dom_acc = accuracy(all_preds.detach().cpu().numpy(), all_dom_labels.detach().cpu().numpy())
			self.optimizers.step(loss)
			end_time = time.time()
			time_meter.update(end_time - start_time)
			loss_meter.update(float(loss))
			acc_meter.update(float(acc))
			dom_acc_meter.update(float(dom_acc))

			if i % self.args.print_freq == 0:
				print(constant)
				# print(all_preds, all_dom_labels)
				log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})\t' \
					'domain accuracy: {dom_acc.val:.3f}({dom_acc.avg:.3f})'.format(
						i, batch_time=time_meter,
						sentim_loss=loss_meter,
						sentim_acc=acc_meter, dom_acc=dom_acc_meter)
				self.logger.info(log_string)
			self.global_steps += 1

		return self.global_steps


class SSL_kbert_Trainer(object):
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, memory_bank, logger, writer=None):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0
		self.memory_bank = memory_bank

	def train_one_epoch(self, device, epoch=0):
		args = self.args
		train_loader = self.train_loader
		model = self.model
		loss_criterion = self.loss_criterion
		optimizers = self.optimizers
		total_steps = self.total_steps
		logger = self.logger

		loss_meter = AverageMeter()
		sentim_loss_meter = AverageMeter()
		ssl_loss_meter = AverageMeter()
		sentim_acc_meter = AverageMeter()
		time_meter = AverageMeter()
		ssl_acc_meter = AverageMeter()
		model.train()
		
		# end_time = time.time()
		for i, (labeled_batch, src_unlabeled_batch, tgt_unlabeled_batch) in enumerate(tqdm(train_loader)):
		# for i, (labeled_batch, src_unlabeled_batch) in enumerate(tqdm(train_loader)):		
			self.optimizers.scheduler_step()
			# print(optimizers.optimizers['bert'].get_lr()[0])
			model.zero_grad()
			if self.args.use_kg:
				tokens_kg1, mask_kg1, pos1, vm1, tokens_org1, mask_org1, ssl_label1, labels = \
					(labeled_batch[k].to(device) for k in ['tokens_kg', 'mask_kg', 'pos', 'vm', 'tokens_org', 'mask_org', 'ssl_label', 'label'])
				tokens_kg2, mask_kg2, pos2, vm2, tokens_org2, mask_org2, ssl_label2 = \
					(src_unlabeled_batch[k].to(device) for k in ['tokens_kg', 'mask_kg', 'pos', 'vm', 'tokens_org', 'mask_org', 'ssl_label'])
				tokens_kg3, mask_kg3, pos3, vm3, tokens_org3, mask_org3, ssl_label3 = \
					(tgt_unlabeled_batch[k].to(device) for k in ['tokens_kg', 'mask_kg', 'pos', 'vm', 'tokens_org', 'mask_org', 'ssl_label'])
			else:
				tokens_kg1, mask_kg1, tokens_org1, mask_org1, ssl_label1, labels = \
					(labeled_batch[k].to(device) for k in ['tokens_kg', 'mask_kg', 'tokens_org', 'mask_org', 'ssl_label', 'label'])
				tokens_kg2, mask_kg2, tokens_org2, mask_org2, ssl_label2 = \
					(src_unlabeled_batch[k].to(device) for k in ['tokens_kg', 'mask_kg', 'tokens_org', 'mask_org', 'ssl_label'])

				# tokens_kg1, mask_kg1, labels = \
				# 	(labeled_batch[k].to(device) for k in ['tokens', 'mask', 'label'])

				tokens_kg3, mask_kg3, tokens_org3, mask_org3, ssl_label3 = \
					(tgt_unlabeled_batch[k].to(device) for k in ['tokens_kg', 'mask_kg', 'tokens_org', 'mask_org', 'ssl_label'])
				pos1, vm1 = None, None
				pos2, vm2 = None, None
				pos3, vm3 = None, None


			tokens_org = torch.cat([tokens_org1, tokens_org2, tokens_org3], dim=0)
			mask_org = torch.cat([mask_org1, mask_org2, mask_org3], dim=0)
			ssl_label = torch.cat([ssl_label1, ssl_label2, ssl_label3], dim=0)	

			start_time = time.time()
			# src labeled data
			logits = model(kg_input=(tokens_kg1, mask_kg1, pos1, vm1), org_input=None, ssl_label=None)

			if self.args.update:
				# src unlabeled data
				logits2 = model(kg_input=(tokens_kg2, mask_kg2, pos2, vm2), org_input=None, ssl_label=None)

				# tgt unlabeled data
				logits3 = model(kg_input=(tokens_kg3, mask_kg3, pos3, vm3), org_input=None, ssl_label=None)
			
			# ssl
			pivot_preds = model(kg_input=None, org_input=(tokens_org, mask_org), ssl_label=ssl_label)

			# print(time.time() - start_time, time.time() - end_time)
			ssl_label = ssl_label.view(-1)
			ssl_label = ssl_label[ssl_label > 0]
			# print(pivot_preds.shape)

			# print(ssl_label.shape)
			# print(tokens_kg1.shape)
			# print('trainer')

			# print(ssl_label)
			# print(labeled_batch['text'])
			# print(src_unlabeled_batch['text'])
			# print(tgt_unlabeled_batch['text'])

			loss, sentim_loss, ssl_loss = loss_criterion(logits, labels, pivot_preds, ssl_label)

			sentim_acc, pred_labels1, conf1 = accuracy(logits.detach().cpu().numpy(), labels.detach().cpu().numpy(), return_pred_and_conf=True) # labeled
			if self.args.update:
				_, pred_labels2, conf2 = accuracy(logits2.detach().cpu().numpy(), None, True)
				_, pred_labels3, conf3 = accuracy(logits3.detach().cpu().numpy(), None, True)
				self.memory_bank.update(labeled_batch['text'], pred_labels1, conf1, 'source', step=False)
				self.memory_bank.update(src_unlabeled_batch['text'], pred_labels2, conf2, 'source', step=False)
				self.memory_bank.update(tgt_unlabeled_batch['text'], pred_labels3, conf3, 'target', step=True)

			ssl_acc = accuracy(pivot_preds.detach().cpu().numpy(), ssl_label.detach().cpu().numpy())

			if i % self.args.balanced_interval == 0:
				# print(i)
				optimizers.step(loss)
			else:
				optimizers.step(ssl_loss)

			end_time = time.time()
			ssl_acc_meter.update(ssl_acc)
			time_meter.update(end_time-start_time)
			loss_meter.update(float(loss))
			sentim_loss_meter.update(float(sentim_loss))
			ssl_loss_meter.update(float(ssl_loss))
			sentim_acc_meter.update(float(sentim_acc))

			if i % self.args.print_freq==0:
				log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'loss: {loss.val:.3f}({loss.avg:.3f})\t' \
					'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'ssl_loss: {ssl_loss.val:.3f}({ssl_loss.avg:.3f})\t' \
					'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})\t' \
					'ssl_accuracy: {ssl_acc.val:.3f}({ssl_acc.avg:.3f})'.format(
						i, batch_time=time_meter, loss=loss_meter,
						sentim_loss=sentim_loss_meter, ssl_loss=ssl_loss_meter,
						sentim_acc=sentim_acc_meter, ssl_acc=ssl_acc_meter)
				# log_string = 'Iteration[{0}]\t' \
				# 	'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
				# 	'loss: {loss.val:.3f}({loss.avg:.3f})\t' \
				# 	'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
				# 	'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})\t'.format(
				# 		i, batch_time=time_meter, loss=loss_meter,
				# 		sentim_loss=sentim_loss_meter,
				# 		sentim_acc=sentim_acc_meter)
				logger.info(log_string)
			self.global_steps += 1
			# end_time = time.time()
			del loss, sentim_loss

		logger.info('-----Training epoch summary------')
		log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'loss: {loss.val:.3f}({loss.avg:.3f})\t' \
					'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, loss=loss_meter,
						sentim_loss=sentim_loss_meter,
						sentim_acc=sentim_acc_meter)
		logger.info(log_string)

		return self.global_steps


class SSL_kbert_DANN_Trainer(object):
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, memory_bank, logger, writer=None):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0
		self.memory_bank = memory_bank

	def train_one_epoch(self, device, epoch=0):
		start_steps = epoch * len(self.train_loader)
		total_steps = self.args.epochs_num * len(self.train_loader)
		args = self.args
		train_loader = self.train_loader
		model = self.model
		loss_criterion = self.loss_criterion
		optimizers = self.optimizers
		total_steps = self.total_steps
		logger = self.logger

		loss_meter = AverageMeter()
		sentim_loss_meter = AverageMeter()
		ssl_loss_meter = AverageMeter()
		sentim_acc_meter = AverageMeter()
		time_meter = AverageMeter()
		ssl_acc_meter = AverageMeter()
		dom_acc_meter = AverageMeter()
		model.train()
		
		# end_time = time.time()
		for i, (labeled_batch, src_unlabeled_batch, tgt_unlabeled_batch) in enumerate(tqdm(train_loader)):	
			self.optimizers.scheduler_step()
			# print(optimizers.optimizers['bert'].get_lr()[0])
			model.zero_grad()
			if self.args.use_kg:
				tokens_kg1, mask_kg1, pos1, vm1, tokens_org1, mask_org1, ssl_label1, labels, dom1 = \
					(labeled_batch[k].to(device) for k in ['tokens_kg', 'mask_kg', 'pos', 'vm', 'tokens_org', 'mask_org', 'ssl_label', 'label', 'domain'])
				tokens_kg2, mask_kg2, pos2, vm2, tokens_org2, mask_org2, ssl_label2, dom2 = \
					(src_unlabeled_batch[k].to(device) for k in ['tokens_kg', 'mask_kg', 'pos', 'vm', 'tokens_org', 'mask_org', 'ssl_label', 'domain'])
				tokens_kg3, mask_kg3, pos3, vm3, tokens_org3, mask_org3, ssl_label3, dom3 = \
					(tgt_unlabeled_batch[k].to(device) for k in ['tokens_kg', 'mask_kg', 'pos', 'vm', 'tokens_org', 'mask_org', 'ssl_label', 'domain'])
			else:
				tokens_kg1, mask_kg1, tokens_org1, mask_org1, ssl_label1, labels = \
					(labeled_batch[k].to(device) for k in ['tokens_kg', 'mask_kg', 'tokens_org', 'mask_org', 'ssl_label', 'label'])
				tokens_kg2, mask_kg2, tokens_org2, mask_org2, ssl_label2 = \
					(src_unlabeled_batch[k].to(device) for k in ['tokens_kg', 'mask_kg', 'tokens_org', 'mask_org', 'ssl_label'])

				# tokens_kg1, mask_kg1, labels = \
				# 	(labeled_batch[k].to(device) for k in ['tokens', 'mask', 'label'])

				tokens_kg3, mask_kg3, tokens_org3, mask_org3, ssl_label3 = \
					(tgt_unlabeled_batch[k].to(device) for k in ['tokens_kg', 'mask_kg', 'tokens_org', 'mask_org', 'ssl_label'])
				pos1, vm1 = None, None
				pos2, vm2 = None, None
				pos3, vm3 = None, None

			tokens_org = torch.cat([tokens_org1, tokens_org2, tokens_org3], dim=0)
			mask_org = torch.cat([mask_org1, mask_org2, mask_org3], dim=0)
			ssl_label = torch.cat([ssl_label1, ssl_label2, ssl_label3], dim=0)	

			# setup hyperparameters
			p = float(i + start_steps) / total_steps
			constant = 2. / (1. + np.exp(-self.args.gamma * p)) - 1

			start_time = time.time()
			# src labeled data
			logits, dom_preds = model(kg_input=(tokens_kg1, mask_kg1, pos1, vm1), constant=constant, org_input=None, ssl_label=None)

			# src unlabeled data
			logits2, dom_preds2 = model(kg_input=(tokens_kg2, mask_kg2, pos2, vm2), constant=constant, org_input=None, ssl_label=None)

			# tgt unlabeled data
			logits3, dom_preds3 = model(kg_input=(tokens_kg3, mask_kg3, pos3, vm3), constant=constant, org_input=None, ssl_label=None)
			
			# ssl
			pivot_preds = model(kg_input=None, org_input=(tokens_org, mask_org), ssl_label=ssl_label)

			ssl_label = ssl_label.view(-1)
			ssl_label = ssl_label[ssl_label > 0]

			all_dom_preds = torch.cat([dom_preds, dom_preds2, dom_preds3], dim=0)
			all_dom_labels = torch.cat([dom1, dom2, dom3], dim=0)
			loss, sentim_loss, ssl_loss, domain_loss = self.loss_criterion(logits, labels, pivot_preds, ssl_label, all_dom_preds, all_dom_labels)
			
			dom_acc = accuracy(all_dom_preds.detach().cpu().numpy(), all_dom_labels.detach().cpu().numpy())
			sentim_acc, pred_labels1, conf1 = accuracy(logits.detach().cpu().numpy(), labels.detach().cpu().numpy(), return_pred_and_conf=True) # labeled
			if self.args.update:
				_, pred_labels2, conf2 = accuracy(logits2.detach().cpu().numpy(), None, True)
				_, pred_labels3, conf3 = accuracy(logits3.detach().cpu().numpy(), None, True)
				self.memory_bank.update(labeled_batch['text'], pred_labels1, conf1, 'source', step=False)
				self.memory_bank.update(src_unlabeled_batch['text'], pred_labels2, conf2, 'source', step=False)
				self.memory_bank.update(tgt_unlabeled_batch['text'], pred_labels3, conf3, 'target', step=True)

			ssl_acc = accuracy(pivot_preds.detach().cpu().numpy(), ssl_label.detach().cpu().numpy())

			optimizers.step(loss)

			end_time = time.time()
			ssl_acc_meter.update(ssl_acc)
			time_meter.update(end_time-start_time)
			loss_meter.update(float(loss))
			sentim_loss_meter.update(float(sentim_loss))
			ssl_loss_meter.update(float(ssl_loss))
			sentim_acc_meter.update(float(sentim_acc))
			dom_acc_meter.update(float(dom_acc))

			if i % self.args.print_freq==0:
				log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'loss: {loss.val:.3f}({loss.avg:.3f})\t' \
					'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'ssl_loss: {ssl_loss.val:.3f}({ssl_loss.avg:.3f})\t' \
					'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})\t' \
					'ssl_accuracy: {ssl_acc.val:.3f}({ssl_acc.avg:.3f})\t' \
					'dom_accuracy: {dom_acc.val:.3f}({dom_acc.avg:.3f})'.format(
						i, batch_time=time_meter, loss=loss_meter,
						sentim_loss=sentim_loss_meter, ssl_loss=ssl_loss_meter,
						sentim_acc=sentim_acc_meter, ssl_acc=ssl_acc_meter, dom_acc=dom_acc_meter)
				logger.info(log_string)
			self.global_steps += 1
			# end_time = time.time()

		logger.info('-----Training epoch summary------')
		log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'loss: {loss.val:.3f}({loss.avg:.3f})\t' \
					'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, loss=loss_meter,
						sentim_loss=sentim_loss_meter,
						sentim_acc=sentim_acc_meter)
		logger.info(log_string)

		return self.global_steps



trainer_factory = {'graph_causal': graph_causal_Trainer,
				'graph': None,
				'causal': causal_Trainer,
				'sentim': sentim_Trainer,
				'base_DA': base_DA_Trainer,
				'kbert_two_stage_da': kbert_two_stage_da_Trainer,
				'kbert_two_stage_sentim': kbert_two_stage_sentim_Trainer,
				'DANN_kbert': DANN_Trainer,
				'SSL_kbert': SSL_kbert_Trainer,
				'SSL_kbert_DANN': SSL_kbert_DANN_Trainer
				}
