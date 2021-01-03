import torch

# cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')


'''
copy from invariant rationalization
'''
def cal_sparsity_loss(z, mask, level):
    """
    Exact sparsity loss in a batchwise sense. 
    Inputs: 
        z -- (batch_size, sequence_length)
        mask -- (batch_size, seq_length)
        level -- sparsity level
    """
    sparsity = torch.sum(z) / torch.sum(mask)
    return torch.abs(sparsity - level)

'''
copy from invariant rationalization
'''
def cal_continuity_loss(z):
    """
    Compute the continuity loss.
    Inputs:     
        z -- (batch_size, sequence_length)
    """
    return torch.mean(torch.abs(z[:, 1:] - z[:, :-1]))

class causal_inference_loss(torch.nn.Module):
    def __init__(self, args):
        super(causal_inference_loss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.sparsity_loss = cal_sparsity_loss
        self.continuity_loss = cal_continuity_loss
        self.sparsity_percentage = args.sparsity_percentage
        self.sparsity_lambda = args.sparsity_lambda
        self.continuity_lambda = args.continuity_lambda
        self.diff_lambda = args.diff_lambda

    def forward(self, sentim, env_enable_sentim, sentim_label, rationale_mask, padding_mask):
        # print(sentim.shape)
        # print(sentim_label.shape)
        # print(sentim_label.dtype)
        # print(sentim.dtype)
        sentim_loss = self.criterion(sentim, sentim_label)
        env_enable_sentim_loss = self.criterion(env_enable_sentim, sentim_label)
        sparsity_loss = self.sparsity_loss(rationale_mask[:,:,0], padding_mask, self.sparsity_percentage)
        continuity_loss = self.continuity_loss(rationale_mask[:,:,0])
        extractor_loss = self.diff_lambda * torch.max((sentim_loss - env_enable_sentim_loss),0)[0] + sentim_loss\
             + self.sparsity_lambda * sparsity_loss + self.continuity_lambda * continuity_loss
        # extractor_loss = self.diff_lambda * torch.max((env_enable_sentim_loss - sentim_loss),0)[0] + env_enable_sentim_loss\
        #    + self.sparsity_lambda * sparsity_loss + self.continuity_lambda * continuity_loss

        return extractor_loss, sentim_loss, env_enable_sentim_loss


class sentim_loss(torch.nn.Module):
    def __init__(self, args):
        super(sentim_loss, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.criterion = torch.nn.NLLLoss()

    def forward(self, logits, labels):
        # softmax_logits = self.softmax(logits)
        loss = self.criterion(self.softmax(logits.view(-1,2)),labels.view(-1))
        return loss

class sentim_loss_ensemble(torch.nn.Module):
    def __init__(self, args):
        super(sentim_loss_ensemble, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, logits, labels):
        return self.criterion(logits, labels)


class cross_entropy_loss(torch.nn.Module):
    def __init__(self):
        super(cross_entropy_loss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, pred, labels):
        loss = self.criterion(pred, labels)
        return loss


class DANN_loss(torch.nn.Module):
    def __init__(self, args):
        super(DANN_loss, self).__init__()
        self.cross_entropy = cross_entropy_loss()

    def forward(self, class_preds, labels, all_preds, all_dom_labels):
        class_loss = self.cross_entropy(class_preds, labels)
        if all_dom_labels is not None:
            domain_loss = self.cross_entropy(all_preds, all_dom_labels)
        else:
            domain_loss = torch.tensor(0.)
        theta = 1
        loss = class_loss + theta * domain_loss
        # print(class_loss.item(), domain_loss.item())
        return loss, class_loss, domain_loss


class SSL_kbert_loss(torch.nn.Module):
    def __init__(self, args):
        super(SSL_kbert_loss, self).__init__()
        self.cross_entropy = cross_entropy_loss()

    def forward(self, class_preds, labels, pivot_preds=None, pivot_labels=None):
        # print(labels)
        # print(pivot_labels)
        class_loss = self.cross_entropy(class_preds, labels)
        if pivot_labels is not None:
            ssl_loss = 0.01 * self.cross_entropy(pivot_preds, pivot_labels)
        else:
            ssl_loss = torch.tensor(0.)
        loss = class_loss + ssl_loss
        return loss, class_loss, ssl_loss


loss_factory = {
    'causal': causal_inference_loss,
    'sentim': sentim_loss,
    'base_DA': sentim_loss,
    'kbert_two_stage_sentim': sentim_loss,
    'kbert_two_stage_da': sentim_loss,
    'DANN_kbert': DANN_loss,
    'SSL_kbert': SSL_kbert_loss
}