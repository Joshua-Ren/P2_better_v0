'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import time
import random
import math
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import sys
sys.path.append("..")
from models import *

def rnd_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ========= Functions about the checkpoint
def save_checkpoint(model, save_path, file_name='test'):
    file_path = os.path.join(save_path, file_name+'.pt')
    torch.save(model.state_dict(), file_path)
    
def get_Alice_Bob_dict(tmp_model):
    from collections import OrderedDict
    alice_dict = OrderedDict()
    bob_dict = OrderedDict()
    for name, param in tmp_model.Alice.named_parameters():
        alice_dict[name] = param
    for name, param in tmp_model.Bob.named_parameters():
        bob_dict[name] = param
    return alice_dict, bob_dict

def load_checkpoint(args, model, ckp_path, which_part='all'):
    '''
        Use this to load params of specific part (Alice, Bob or all),
        from ckp to model.
    '''
    if which_part.lower()=='all':
        model.load_state_dict(torch.load(ckp_path))
    elif which_part.lower()=='alice':
        tmp_model = get_init_net(args)
        tmp_model.load_state_dict(torch.load(ckp_path))
        alice_dict, _ = get_Alice_Bob_dict(tmp_model)
        model.Alice.load_state_dict(alice_dict,strict=False)
    elif which_part.lower()=='bob':
        tmp_model = get_init_net(args)
        tmp_model.load_state_dict(torch.load(ckp_path))
        _, bob_dict = get_Alice_Bob_dict(tmp_model)
        model.Bob.load_state_dict(bob_dict,strict=False)
    else:
        print('which_part must be alice, bob, or all')      


# =========== wandb functions =================
def wandb_init(proj_name='test', run_name=None, config_args=None):
    wandb.init(
        project=proj_name,
        config={})
    if config_args is not None:
        wandb.config.update(config_args)
    if run_name is not None:
        wandb.run.name=run_name
        return run_name
    else:
        return wandb.run.name

def wandb_record_results(results, epoch):
  for key in results.keys():
    wandb.log({key:results[key][-1]})
  wandb.log({'epoch':epoch})

# ======== Get Model ===================
def get_init_net(args, force_type=None):
    if force_type is None:
        net_type = args.model
    else:
        net_type = force_type

    if net_type=='resnet18':
        net = ResNet18(args.nb_classes, AB_split=args.AB_split)
    elif net_type=='resnet50':
        net = ResNet50(args.nb_classes, AB_split=args.AB_split)
    elif net_type=='efficientb3':
        from efficientnet_pytorch import EfficientNet
        net = EfficientNet.from_pretrained('efficientnet-b3', num_classes=args.nb_class)
    else:
        print('net structure not supported, only support resnet18, resnet50, mobile, vgg, efficientb3')
    return net

def get_optimizer(model, args):
    if args.optim_type.lower()=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=args.weight_decay,nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ft_epochs, eta_min=args.min_lr)
    elif args.optim_type.lower()=='adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ft_epochs, eta_min=args.min_lr)
    return optimizer, scheduler

# =========== Track the results ==================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

"""
def save_checkpoint(model, save_path, file_name='test', which_part='all'):
    file_path = os.path.join(save_path, file_name+'.pt')
    if which_part.lower()=='alice':
        torch.save(model.Alice.state_dict(), file_path)
    elif which_part.lower()=='bob':
        torch.save(model.Bob.state_dict(), file_path)
    elif which_part.lower()=='all':
        torch.save(model.state_dict(), file_path)
    else:
        torch.save(model.state_dict(), file_path)
"""