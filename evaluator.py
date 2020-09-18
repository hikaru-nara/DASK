import torch 
from utils.utils import accuracy
from utils.utils import AverageMeter

evaluator_factory = {
        'causal': causal_Evaluator 
        }

class causal_Evaluator(object):
    '''
    @Tian Li
    '''
    def __init__(self, eval_loader, model, loss_criterion, logger):
        self.eval_loader = eval_loader
        self.model = model
        self.loss_criterion = loss_criterion
        self.logger = logger

    def evaluate_one_epoch(self):
        eval_loader = self.eval_loader
        model = self.model
        loss_criterion = self.loss_criterion
        logger = self.logger
        sentim_acc_meter = AverageMeter()
        dom_acc_meter = AverageMeter()
        sentim_loss_meter = AverageMeter()
        dom_loss_meter = AverageMeter()
        model.eval()

        logger.info('-------Start evaluation------')
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                tokens, masks, positions, vms, augs, labels = batch
                sentim_probs, env_enable_sentim_probs, rationale_mask = model(tokens, masks, positions, vms, augs)
                _, sentim_loss, env_enable_sentim_loss = loss_criterion(sentim_probs, \
                    env_enable_sentim_probs, labels, rationale_mask, masks)

                sentim_acc = accuracy(sentim_probs, labels)
                env_enable_sentim_acc = accuracy(env_enable_sentim_probs, labels)

                sentim_acc_meter.update(sentim_acc, n=batch.shape[0])
                dom_acc_meter.update(dom_acc, n=batch.shape[0])
                sentim_loss_meter.update(sentim_loss, n=batch.shape[0])
                dom_loss_meter.update(dom_loss, n=batch.shape[0])

                if i % args.print_freq==0:
                    log_string = 'Iteration[{0}]\t' \
                        'domain_loss {dom_loss.val:.3f}({dom_loss.avg:.3f})\t' \
                        'sentiment_loss {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
                        'domain_accuracy {dom_acc.val:.3f}({dom_acc.avg:.3f})\t' \
                        'sentiment_accuracy {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
                            i, dom_loss=dom_loss_meter, 
                            sentim_loss=sentim_loss_meter, dom_acc=dom_acc_meter, 
                            sentim_acc=sentim_acc_meter)
                    logger.info(log_string)
        return sentim_acc_meter.avg

