import torch 
from utils.utils import AverageMeter
from utils.utils import accuracy, consistence, get_optimizer
import time
from tqdm import tqdm

def compare(param_group, model):
	print('compare')
	m = model.children().__next__()
	param_group_new = [c.parameters() for c in m.children()]
	print(len(param_group_new))
	for i,(p1,p2) in enumerate(zip(param_group, param_group_new)):
		# print(len(p1))
		for j,(pi1, pi2) in enumerate(zip(p1, p2)):
			print(i, j, torch.eq(pi1, pi2).all())
		# print(len(p1), len(p2))
		# torch.all(torch.eq(p1,p2))

def compare2(param_group, opt):
	print('compare2')
	param_o = opt.param_groups[0]['params']
	for p1, p2 in zip(param_group, param_o):
		print(torch.eq(p1,p2).all())

def compare_param(param, param_new):
	print('compare_param')
	flag = True
	flag_is = True
	for p1, p2 in zip(param, param_new):
		flag = flag and torch.eq(p1,p2).all().item()
		flag_is = flag_is and (p1 is p2)
	print(flag, flag_is)

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
		env_enable_sentim_loss_meter = AverageMeter()
		env_enable_sentim_acc_meter = AverageMeter()

		model.train()
		for i, batch_data in enumerate(train_loader):
			model.zero_grad()
			tokens, masks, positions, vms, augs, labels = batch_data['token'], batch_data['mask'], \
				batch_data['pos'], batch_data['vm'], batch_data['aug'], batch_data['label']
				
			labels = labels.long().to(device)
			masks = masks.long().to(device)
			positions = positions.long().to(device)
			vms = vms.long().to(device)
			augs = augs.to(device)
			tokens = tokens.long().to(device)
			start_time = time.time()

			sentim_probs, env_enable_sentim_probs, rationale_mask = model(tokens, masks, positions, vms, augs)
			
			extractor_loss, sentim_loss, env_enable_sentim_loss = loss_criterion(sentim_probs, \
					env_enable_sentim_probs, labels, rationale_mask, masks)

			sentim_acc = accuracy(sentim_probs, labels)
			env_enable_sentim_acc = accuracy(env_enable_sentim_probs, labels)

			optimizers.step([extractor_loss, sentim_loss, env_enable_sentim_loss],i)

			end_time = time.time()
			time_meter.update(end_time-start_time)
			sentim_loss_meter.update(float(sentim_loss))
			sentim_acc_meter.update(float(sentim_acc))
			env_enable_sentim_acc_meter.update(float(env_enable_sentim_acc))
			env_enable_sentim_loss_meter.update(float(env_enable_sentim_loss))

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



trainer_factory = {'graph_causal': graph_causal_Trainer,
				'graph': None,
				'causal': causal_Trainer,
				'sentim': sentim_Trainer,
				'base_DA': base_DA_Trainer}