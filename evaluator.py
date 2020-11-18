import torch 
from utils.utils import accuracy
from utils.utils import AverageMeter
from tqdm import tqdm

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
                
                tokens, masks, positions, vms, labels = batch_data['tokens'], batch_data['mask'], \
                    batch_data['pos'], batch_data['vm'], batch_data['label']
                labels = labels.long().to(device)
                masks = masks.long().to(device)
                # positions = positions.long().to(device)
                # vms = vms.long().to(device)
                tokens = tokens.long().to(device)

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

    def eval_one_epoch(self, device):
        eval_loader = self.eval_loader
        model = self.model
        loss_criterion = self.loss_criterion
        logger = self.logger
        sentim_acc_meter = AverageMeter()
        sentim_loss_meter = AverageMeter()
        env_enable_sentim_loss_meter = AverageMeter()
        env_enable_sentim_acc_meter = AverageMeter()
        model.eval()

        logger.info('-------Start evaluation-------')
        with torch.no_grad():
            for i, batch_data in enumerate(eval_loader):
                
                tokens, masks, positions, vms, augs, labels = batch_data['token'], batch_data['mask'], \
                    batch_data['pos'], batch_data['vm'], batch_data['aug'], batch_data['label']
                labels = labels.long().to(device)
                masks = masks.to(device)
                positions = positions.to(device)
                vms = vms.long().to(device)
                augs = augs.to(device)

                batch_size = labels.shape[0]
                sentim_probs, env_enable_sentim_probs, rationale_mask = model(tokens, masks, positions, vms, augs)
                _, sentim_loss, env_enable_sentim_loss = loss_criterion(sentim_probs, \
                    env_enable_sentim_probs, labels, rationale_mask, masks)

                sentim_acc = accuracy(sentim_probs, labels)
                env_enable_sentim_acc = accuracy(env_enable_sentim_probs, labels)


                sentim_acc_meter.update(sentim_acc, n=batch_size)
                sentim_loss_meter.update(sentim_loss, n=batch_size)
                env_enable_sentim_acc_meter.update(env_enable_sentim_acc)
                env_enable_sentim_loss_meter.update(env_enable_sentim_loss)

                if i % self.args.print_freq==0:
                    log_string = 'Iteration[{0}]\t' \
                    'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
                    'env_enable_sentiment_loss: {env_enable_sentim_loss.val:.3f}({env_enable_sentim_loss.avg:.3f})' \
                    'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})' \
                    'env_enable_sentiment_acc: {env_enable_sentim_acc.val:.3f}({env_enable_sentim_acc.avg:.3f})'.format(
                        i, env_enable_sentim_loss=env_enable_sentim_loss_meter,
                        sentim_loss=sentim_loss_meter, env_enable_sentim_acc=env_enable_sentim_acc_meter,
                        sentim_acc=sentim_acc_meter)
                    logger.info(log_string)
            logger.info('-----Evaluation epoch summary------')
            log_string = 'Iteration[{0}]\t' \
                    'sentiment_loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
                    'env_enable_sentiment_loss: {env_enable_sentim_loss.val:.3f}({env_enable_sentim_loss.avg:.3f})' \
                    'sentiment_accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})' \
                    'env_enable_sentiment_acc: {env_enable_sentim_acc.val:.3f}({env_enable_sentim_acc.avg:.3f})'.format(
                        i, env_enable_sentim_loss=env_enable_sentim_loss_meter,
                        sentim_loss=sentim_loss_meter, env_enable_sentim_acc=env_enable_sentim_acc_meter,
                        sentim_acc=sentim_acc_meter)
            logger.info(log_string)
        return sentim_acc_meter.avg


evaluator_factory = {
        'causal': causal_Evaluator ,
        'sentim': sentim_Evaluator
        }