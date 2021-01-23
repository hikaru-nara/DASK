import torch 
from utils.utils import accuracy
from utils.utils import AverageMeter
from utils.utils import save_attention_mask, save_fail_cases
from tqdm import tqdm
import time
from transformers import BertTokenizer



class kbert_two_stage_Evaluator(object):
    '''
    @tian Li
    '''
    def __init__(self, args, eval_loader, model, loss_criterion, logger):
        self.args = args
        self.eval_loader = eval_loader
        self.model = model
        self.loss_criterion = loss_criterion
        self.logger = logger

    def eval_one_epoch(self, device):
        eval_loader = self.eval_loader
        model = self.model
        loss_criterion = self.loss_criterion
        logger = self.logger
        sentim_acc_meter = AverageMeter()
        sentim_loss_meter = AverageMeter()

        model.eval()


        logger.info('-------Start evaluation-------')
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(eval_loader)):
                
                tokens_kg, mask_kg, labels, tokens_org, mask_org = batch_data['tokens_kg'], batch_data['mask_kg'], \
                 batch_data['label'], batch_data['tokens_org'], batch_data['mask_org']

                if self.args.use_kg:
                    positions, vms = batch_data['pos'], batch_data['vm']
                else:
                    positions, vms = None, None
                batch_size = labels.shape[0]
                labels = labels.long().to(device)
                mask_kg = mask_kg.long().to(device)
                # positions = positions.long().to(device)
                # vms = vms.long().to(device)
                tokens_org, mask_org = tokens_org.long().to(device), mask_org.long().to(device)
                tokens_kg = tokens_kg.long().to(device)

                sentim_probs = model(tokens_kg, tokens_org, mask_kg, mask_org, positions, vms)
                sentim_loss = loss_criterion(sentim_probs, labels)

                sentim_acc = accuracy(sentim_probs.detach().cpu().numpy(), labels.detach().cpu().numpy())

                sentim_acc_meter.update(sentim_acc, n=batch_size)
                sentim_loss_meter.update(sentim_loss, n=batch_size)

                if i % self.args.print_freq==0:
                    log_string = 'Iteration[{0}]\t' \
                    'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
                    'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
                        i, 
                        sentim_loss=sentim_loss_meter,
                        sentim_acc=sentim_acc_meter)
                    logger.info(log_string)

            logger.info('-----Evaluation epoch summary------')
            log_string = 'Iteration[{0}]\t' \
                    'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
                    'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
                        i, 
                        sentim_loss=sentim_loss_meter,
                        sentim_acc=sentim_acc_meter)
            logger.info(log_string)
        return sentim_acc_meter.avg


class SSL_kbert_Evaluator(object):
    '''
    @tian Li
    '''
    def __init__(self, args, eval_loader, model, loss_criterion, logger):
        self.args = args
        self.eval_loader = eval_loader
        self.model = model
        self.loss_criterion = loss_criterion
        self.logger = logger

    def eval_one_epoch(self, device):
        eval_loader = self.eval_loader
        model = self.model
        loss_criterion = self.loss_criterion
        logger = self.logger
        sentim_acc_meter = AverageMeter()
        sentim_loss_meter = AverageMeter()

        model.eval()


        logger.info('-------Start evaluation-------')
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(eval_loader)):
                tokens_kg, mask_kg, labels, tokens_org, mask_org = batch_data['tokens_kg'], batch_data['mask_kg'], \
                 batch_data['label'], batch_data['tokens_org'], batch_data['mask_org']

                if self.args.use_kg:
                    positions, vms = batch_data['pos'], batch_data['vm']
                else:
                    positions, vms = None, None
                batch_size = labels.shape[0]
                labels = labels.long().to(device)
                mask_kg = mask_kg.long().to(device)
                # positions = positions.long().to(device)
                # vms = vms.long().to(device)
                tokens_org, mask_org = tokens_org.long().to(device), mask_org.long().to(device)
                tokens_kg = tokens_kg.long().to(device)

                logits = model(kg_input=(tokens_kg, mask_kg, positions, vms))

                sentim_loss, _, _ = loss_criterion(logits, labels)
                
                sentim_acc = accuracy(logits.detach().cpu().numpy(), labels.detach().cpu().numpy())

                sentim_acc_meter.update(sentim_acc, n=batch_size)
                sentim_loss_meter.update(sentim_loss, n=batch_size)

                if i % self.args.print_freq==0:
                    log_string = 'Iteration[{0}]\t' \
                    'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
                    'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
                        i, 
                        sentim_loss=sentim_loss_meter,
                        sentim_acc=sentim_acc_meter)
                    logger.info(log_string)

            logger.info('-----Evaluation epoch summary------')
            log_string = 'Iteration[{0}]\t' \
                    'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
                    'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
                        i, 
                        sentim_loss=sentim_loss_meter,
                        sentim_acc=sentim_acc_meter)
            logger.info(log_string)
        return sentim_acc_meter.avg
        

class sentim_Evaluator(object):
    '''
    @tian Li
    '''
    def __init__(self, args, eval_loader, model, loss_criterion, logger):
        self.args = args
        self.eval_loader = eval_loader
        self.model = model
        self.loss_criterion = loss_criterion
        self.logger = logger

    def eval_one_epoch(self, device):
        eval_loader = self.eval_loader
        model = self.model
        loss_criterion = self.loss_criterion
        logger = self.logger
        sentim_acc_meter = AverageMeter()
        sentim_loss_meter = AverageMeter()

        model.eval()


        logger.info('-------Start evaluation-------')
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(eval_loader)):
                
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

                batch_size = labels.shape[0]
                if self.args.save_attention_mask and i==0:
                    sentim_probs, attentions = model(tokens, masks, pos=positions, vm=vms, output_attention=True, only_first_vm=self.args.only_first_vm)
                    attentions = [a.detach().cpu().numpy() for a in attentions]
                    # save_attention_mask(attentions, batch_data['text'], batch_data['pos'], batch_data['tokens'], self.args.log_dir)

                else:
                    sentim_probs = model(tokens, masks, pos=positions, vm=vms, output_attention=False, only_first_vm=self.args.only_first_vm)
                sentim_loss = loss_criterion(sentim_probs, labels)

                sentim_acc = accuracy(sentim_probs.detach().cpu().numpy(), labels.detach().cpu().numpy())
                if self.args.save_fail_cases:
                    if save_fail_cases(sentim_probs.detach().cpu().numpy(), labels.detach().cpu().numpy(), 
                        attentions, batch_data['text'], batch_data['pos'], batch_data['tokens'], self.args.log_dir):
                        break

                sentim_acc_meter.update(sentim_acc, n=batch_size)
                sentim_loss_meter.update(sentim_loss, n=batch_size)

                if i % self.args.print_freq==0:
                    log_string = 'Iteration[{0}]\t' \
                    'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
                    'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
                        i, 
                        sentim_loss=sentim_loss_meter,
                        sentim_acc=sentim_acc_meter)
                    logger.info(log_string)

            logger.info('-----Evaluation epoch summary------')
            log_string = 'Iteration[{0}]\t' \
                    'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
                    'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
                        i, 
                        sentim_loss=sentim_loss_meter,
                        sentim_acc=sentim_acc_meter)
            logger.info(log_string)
        return sentim_acc_meter.avg


class DANN_Evaluator(object):
    def __init__(self, args, eval_loader, model, loss_criterion, logger, writer=None, tokenizer=None):
        self.args = args
        self.eval_loader = eval_loader
        self.model = model
        self.loss_criterion = loss_criterion
        self.logger = logger
        self.global_steps = 0
        self.tokenizer = tokenizer

    def eval_one_epoch(self, device, epoch=0):
        self.logger.info('-------Start evaluation-------')
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        time_meter = AverageMeter()
        self.model.eval()
        for i, labeled_data in enumerate(self.eval_loader):
            # print(i)
            # print(labeled_data)
            # print(optimizers.optimizers['bert'].get_lr()[0])
            text, input_ids, masks, labels = labeled_data['text'], labeled_data['tokens'], \
                                             labeled_data['mask'], labeled_data['label']
            input_ids, masks, labels = input_ids.to(device), masks.to(device), labels.to(device)
            start_time = time.time()
            class_preds = self.model(input_ids, masks)
            loss = self.loss_criterion.cross_entropy(class_preds, labels)
            acc = accuracy(class_preds.detach().cpu().numpy(), labels.detach().cpu().numpy())
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
        self.logger.info('evaluation acc: {0:.4f}'.format(acc_meter.avg))
        return acc_meter.avg


class base_DA_Evaluator(object):
    '''
    @tian Li
    '''
    def __init__(self, args, eval_loader, model, loss_criterion, logger):
        self.args = args
        self.eval_loader = eval_loader
        self.model = model
        self.loss_criterion = loss_criterion
        self.logger = logger

    def eval_one_epoch(self, device):
        eval_loader = self.eval_loader
        model = self.model
        loss_criterion = self.loss_criterion
        logger = self.logger
        sentim_acc_meter = AverageMeter()
        sentim_loss_meter = AverageMeter()

        model.eval()


        logger.info('-------Start evaluation-------')
        with torch.no_grad():
            for i, (labeled_batch, unlabeled_batch) in enumerate(eval_loader):
                
                tokens, masks, labels = labeled_batch['tokens'].long().to(device), labeled_batch['mask'].long().to(device), \
                 labeled_batch['label'].long().to(device)
                # tokens_u, masks_u, labels_u = unlabeled_batch['tokens'].long().to(device), unlabeled_batch['mask'].long().to(device), \
                #    unlabeled_batch['label'].long().to(device)
                if self.args.use_kg:
                    positions, vms = labeled_batch['pos'].long().to(device), labeled_batch['vm'].long().to(device)
                    # positions_u, vms_u = unlabeled_batch['pos'].long().to(device), unlabeled_batch['vm'].long().to(device)
                else:
                    positions, vms, positions_u, vms_u = None, None, None, None

                batch_size = labels.shape[0]
                sentim_probs = model(tokens, masks, positions, vms)
                sentim_loss = loss_criterion(sentim_probs, labels)

                sentim_acc = accuracy(sentim_probs.detach().cpu().numpy(), labels.detach().cpu().numpy())

                sentim_acc_meter.update(sentim_acc, n=batch_size)
                sentim_loss_meter.update(sentim_loss, n=batch_size)

                if i % self.args.print_freq==0:
                    log_string = 'Iteration[{0}]\t' \
                    'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
                    'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
                        i, 
                        sentim_loss=sentim_loss_meter,
                        sentim_acc=sentim_acc_meter)
                    logger.info(log_string)

            logger.info('-----Evaluation epoch summary------')
            log_string = 'Iteration[{0}]\t' \
                    'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
                    'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
                        i, 
                        sentim_loss=sentim_loss_meter,
                        sentim_acc=sentim_acc_meter)
            logger.info(log_string)
        return sentim_acc_meter.avg



class causal_Evaluator(object):
    '''
    @Tian Li
    '''
    def __init__(self, args, eval_loader, model, loss_criterion, logger):
        self.args = args
        self.eval_loader = eval_loader
        self.model = model
        self.loss_criterion = loss_criterion
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def eval_one_epoch(self, device):
        eval_loader = self.eval_loader
        model = self.model
        loss_criterion = self.loss_criterion
        logger = self.logger
        sentim_acc_meter = AverageMeter()
        sentim_loss_meter = AverageMeter()
        env_enable_sentim_loss_meter = AverageMeter()
        env_enable_sentim_acc_meter = AverageMeter()
        select_bias = 0
        total_num = 0    
        model.eval()

        logger.info('-------Start evaluation-------')
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(eval_loader)):
                
                tokens, masks, envs, labels = batch_data['tokens'], batch_data['mask'], \
                    batch_data['env'], batch_data['label']
                if self.args.use_kg:
                    positions, vms = batch_data['pos'], batch_data['vm']
                    positions = positions.long().to(device)
                    vms = vms.long().to(device)
                else:
                    positions, vms = None, None
                labels = labels.long().to(device)
                masks = masks.to(device)
                # positions = positions.to(device)
                # vms = vms.long().to(device)
                envs = envs.to(device)
                tokens = tokens.long().to(device)

                batch_size = labels.shape[0]
                sentim_probs, env_enable_sentim_probs, rationale_mask = model(tokens, masks, envs, positions, vms)
                _, sentim_loss, env_enable_sentim_loss = loss_criterion(sentim_probs, \
                    env_enable_sentim_probs, labels, rationale_mask, masks)

                sentim_acc = accuracy(sentim_probs.detach().cpu().numpy(), labels.detach().cpu().numpy())
                env_enable_sentim_acc = accuracy(env_enable_sentim_probs.detach().cpu().numpy(), labels.detach().cpu().numpy())
                select_bias += torch.sum(rationale_mask[:, 1, 0]) # selecting the leading . or ,
                total_num += batch_data['tokens'].shape[0]

                sentim_acc_meter.update(sentim_acc, n=batch_size)
                sentim_loss_meter.update(sentim_loss, n=batch_size)
                env_enable_sentim_acc_meter.update(env_enable_sentim_acc)
                env_enable_sentim_loss_meter.update(env_enable_sentim_loss)

                if i % self.args.print_freq==0:
                    log_string = 'Iteration[{0}]\t' \
                    'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
                    'env_enable_sentiment_loss: {env_enable_sentim_loss.val:.3f}({env_enable_sentim_loss.avg:.3f})\t' \
                    'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})\t' \
                    'env_enable_sentiment_acc: {env_enable_sentim_acc.val:.3f}({env_enable_sentim_acc.avg:.3f})'.format(
                        i, env_enable_sentim_loss=env_enable_sentim_loss_meter,
                        sentim_loss=sentim_loss_meter, env_enable_sentim_acc=env_enable_sentim_acc_meter,
                        sentim_acc=sentim_acc_meter)
                    logger.info(log_string)

                    # display_rational_log_string = ''
                    raw_text = batch_data['text'][0]
                    rationale_tokens = tokens[0,:] * rationale_mask[0,:,0]
                    label = labels[0]

                    rationale_text = self.tokenizer.decode(rationale_tokens)
                    print('raw_text: {}'.format(raw_text))
                    print('rationale_text: {}'.format(rationale_text))
                    print('polarity: {}'.format(label))
            logger.info('-----Evaluation epoch summary------')
            log_string = 'Iteration[{0}]\t' \
                    'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
                    'env_enable_sentiment_loss: {env_enable_sentim_loss.val:.3f}({env_enable_sentim_loss.avg:.3f})\t' \
                    'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})\t' \
                    'env_enable_sentiment_acc: {env_enable_sentim_acc.val:.3f}({env_enable_sentim_acc.avg:.3f})\t' \
                    'select_bias_acc: {select_bias_acc:.3f}'.format(
                        i, env_enable_sentim_loss=env_enable_sentim_loss_meter,
                        sentim_loss=sentim_loss_meter, env_enable_sentim_acc=env_enable_sentim_acc_meter,
                        sentim_acc=sentim_acc_meter, select_bias_acc=select_bias/total_num)
            logger.info(log_string)
        return sentim_acc_meter.avg


evaluator_factory = {
        'causal': causal_Evaluator,
        'sentim': sentim_Evaluator,
        'base_DA': sentim_Evaluator,
        'kbert_two_stage_sentim': kbert_two_stage_Evaluator,
        'kbert_two_stage_da': kbert_two_stage_Evaluator,
        'DANN_kbert': DANN_Evaluator,
        'SSL_kbert': SSL_kbert_Evaluator,
        'SSL_kbert_DANN': SSL_kbert_Evaluator,
        'masked_SSL_kbert': SSL_kbert_Evaluator
        }