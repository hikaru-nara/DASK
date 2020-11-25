import torch
from uer.utils.optimizers import BertAdam
from utils.utils import get_optimizer
from torch.optim import Optimizer
from transformers import AdamW, get_linear_schedule_with_warmup

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




class sentim_optimizer(object):
    def __init__(self, args, model, total_steps):
        self.args = args
        self.optimizers = {}
        module_dict = model.named_children()

        for name, model in module_dict:
            if 'bert' in name:
                if args.freeze_bert:
                    continue
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]
            self.optimizers[name] = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=total_steps)
            # if 'bert' in name:
            #     if args.freeze_bert:
            #         continue
            #     param_optimizer = list(model.named_parameters())
            #     no_decay = ['bias', 'gamma', 'beta']
            #     optimizer_grouped_parameters = [
            #                 {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            #                 {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            #     ]
            #     self.optimizers[name] = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=total_steps)
            # else:
            #     self.optimizers[name] = get_optimizer(args, model.parameters(), lr=lr.get(name, default_lr))
        self.round = [[0,1,2],[3,4],[5,6]]
        self.term = 7
        assert len(self.optimizers) == 2-args.freeze_bert

    def __len__(self):
        return 1

    def step(self, loss, step):
        path = step % 7
        loss.backward()
        for name, opt in self.optimizers.items():
            opt.step()
        # self.optimizer.step()


class da_optimizer(object):
    def __init__(self, args, model, total_steps):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=args.warmup*total_steps,
                                                         num_training_steps=total_steps)

    def __len__(self):
        return 1

    def step(self, loss):
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        self.optimizer.step()

    def scheduler_step(self):
        self.scheduler.step()



class causal_optimizer(object):
    def __init__(self, args, model, total_steps):
        self.args = args
        lr = self.args.lr
        default_lr = lr['default']
        self.optimizers = {}
        module_dict = model.named_children()
        self.schedulers = {}
        for name, model in module_dict:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]
            self.optimizers[name] = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=total_steps)
            self.schedulers[name] = get_linear_schedule_with_warmup(self.optimizers[name],
                                                         num_warmup_steps=args.warmup*total_steps,
                                                         num_training_steps=total_steps)
            # if 'bert' in name:
            #     param_optimizer = list(model.named_parameters())
            #     no_decay = ['bias', 'gamma', 'beta']
            #     optimizer_grouped_parameters = [
            #                 {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            #                 {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            #     ]
            #     self.optimizers[name] = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=total_steps)
            # else:
            #     self.optimizers[name] = get_optimizer(args, model.parameters(), lr=lr.get(name, default_lr))
        self.round = [[0,1,2],[3,4],[5,6]]
        self.term = 7
        assert len(self.optimizers)==4

    def __len__(self):
        return len(self.optimizers)

    def step(self, loss, step):
        path = step%self.term
        # self.optimizers['bert'].step()
        # max_grad(self.optimizers['bert'])
        # self.optimizers['al'].step()
        # max_grad(self.optimizers['al'])
        # self.optimizers['sc'].step()
        # max_grad(self.optimizers['sc'])
        if path in [0,1,2]:
          loss[0].backward()
          self.optimizers['bert'].step()
          # self.optimizers['al'].step()
        elif path in [3,4]:
          loss[1].backward()
          self.optimizers['sc'].step()
        else:
          loss[2].backward()
          self.optimizers['env_sc'].step()

    def scheduler_step(self):
        for k in self.schedulers:
            self.schedulers[k].step()


# class sentim_optimizer(object):
#     def __init__(self, args, module_dict, total_steps):
#         self.args = args
#         lr = self.args.lr
#         default_lr = lr['default']
#         self.optimizers = {}
#         for name, model in module_dict:
#             if 'bert' in name:
#                 param_optimizer = list(model.named_parameters())
#                 no_decay = ['bias', 'gamma', 'beta']
#                 optimizer_grouped_parameters = [
#                             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
#                             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
#                 ]
#                 self.optimizers[name] = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=-1, t_total=total_steps)
#             else:
#                 self.optimizers[name] = get_optimizer(args, model.parameters(), lr=lr.get(name, default_lr))

#     def __len__(self):
#         return len(self.optimizers)

#     def step(self, loss, step):
#         # path = step%self.term
#         loss.backward()
#         self.optimizers['bert'].step()
#         self.optimizers['sc'].step()


optimizer_factory ={
    'causal': causal_optimizer,
    'sentim': da_optimizer,
    'base_DA': da_optimizer,
    'kbert_two_stage_sentim': da_optimizer
}