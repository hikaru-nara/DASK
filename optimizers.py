import torch
from uer.utils.optimizers import BertAdam
from utils.utils import get_optimizer
from torch.optim import Optimizer


def max_grad(opt):
    print('max_grad')
    for i, param_group in enumerate(opt.param_groups):
        params = param_group['params']
        print('group ',i)
        for p in params:
            print(torch.max(p.grad).item())


class assemble_causal_optimizer(Optimizer):
    def __init__(self, args, module_dict, total_steps):
        param_groups = []
        for name, model in module_dict:
            pass


class causal_optimizer(object):
    def __init__(self, args, module_dict, total_steps):
        self.args = args
        lr = self.args.lr
        default_lr = lr['default']
        self.optimizers = {}
        for name, model in module_dict:
            if 'bert' in name:
                param_optimizer = list(model.named_parameters())
                no_decay = ['bias', 'gamma', 'beta']
                optimizer_grouped_parameters = [
                            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
                ]
                self.optimizers[name] = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=-1, t_total=total_steps)
            else:
                self.optimizers[name] = get_optimizer(args, model.parameters(), lr=lr.get(name, default_lr))
        self.round = [[0,1,2],[3,4],[5,6]]
        self.term = 7
        assert len(self.optimizers)==4

    def __len__(self):
        return len(self.optimizers)

    def step(self, loss, step):
        path = step%self.term
        loss[1].backward()
        self.optimizers['bert'].step()
        # max_grad(self.optimizers['bert'])
        self.optimizers['al'].step()
        # max_grad(self.optimizers['al'])
        self.optimizers['sc'].step()
        # max_grad(self.optimizers['sc'])
        # if path in [0,1,2]:
        #   loss[0].backward()
        #   self.optimizers['bert'].step()
        #   self.optimizers['al'].step()
        # elif path in [3,4]:
        #   loss[1].backward()
        #   self.optimizers['sc'].step()
        # else:
        #   loss[2].backward()
        #   self.optimizers['env_sc'].step()


class sentim_optimizer(object):
    def __init__(self, args, module_dict, total_steps):
        self.args = args
        lr = self.args.lr
        default_lr = lr['default']
        self.optimizers = {}
        for name, model in module_dict:
            if 'bert' in name:
                param_optimizer = list(model.named_parameters())
                no_decay = ['bias', 'gamma', 'beta']
                optimizer_grouped_parameters = [
                            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
                ]
                self.optimizers[name] = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=-1, t_total=total_steps)
            else:
                self.optimizers[name] = get_optimizer(args, model.parameters(), lr=lr.get(name, default_lr))

    def __len__(self):
        return len(self.optimizers)

    def step(self, loss, step):
        # path = step%self.term
        loss.backward()
        self.optimizers['bert'].step()
        self.optimizers['sc'].step()


optimizer_factory ={
    'causal': causal_optimizer,
    'sentim': sentim_optimizer
}