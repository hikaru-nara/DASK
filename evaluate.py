import torch 
from utils.utils import AverageMeter

def evaluate_one_epoch(args, eval_loader, model, loss_criterion, logger):
    sentim_acc_meter = AverageMeter()
    dom_acc_meter = AverageMeter()
    sentim_loss_meter = AverageMeter()
    dom_loss_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            tokens, masks, positions, vms, labels, doms, _ = batch
            sentim_logits, dom_logits = model(tokens, masks, positions, vms)
            sentim_loss = loss_criterion(sentim_logits, labels)
            dom_loss = loss_criterion(dom_logits, doms)

            sentim_acc = accuracy(sentim_logits, labels)
            dom_acc = accuracy(dom_logits, doms)

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



def accuracy(batch_logits, batch_labels):
    batch_size = batch_labels.shape[0]
    # pred = torch.eq(batch_logits, torch.max(batch_logits, -1, keepdim=-1))[:,0]
    pred = torch.argmax(batch_logits, -1)
    correct = torch.sum(torch.eq(pred, batch_labels))
    accuracy = correct/batch_size
    return accuracy