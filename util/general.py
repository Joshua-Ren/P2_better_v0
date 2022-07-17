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
from models.resnet import ResNet50
from models.resnet_cifar import ResNet18

def get_bob_grad_norm(args, model):
    tmp_pgrad_bob = torch.tensor([],requires_grad=False).cuda()
    if args.model in ['resnet18', 'resnet50']:
        for name, params in model.Bob.named_parameters():
            tmp_pgrad_bob = torch.cat((tmp_pgrad_bob,params.grad.reshape(-1,1)),axis=0)
        return torch.norm(tmp_pgrad_bob).cpu()
    elif args.model in ['vit16']:
        for name, params in model.head.named_parameters():
            tmp_pgrad_bob = torch.cat((tmp_pgrad_bob,params.grad.reshape(-1,1)),axis=0)
        return torch.norm(tmp_pgrad_bob).cpu()  

def args_get_class(args):
    if args.dataset=='cifar10' or args.dataset=='stl10':
        args.nb_classes=10
    elif args.dataset=='cifar100':
        args.nb_classes=100
    elif args.dataset[:6]=='domain':
        args.nb_classes=200
    return args

def rnd_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def sort_files(file_list):
    index_list = []
    for f in file_list:
        bob_ep = f.split('.')[0].split('_')[-1]
        index_list.append(int(bob_ep))
    sort_mask = np.argsort(np.array(index_list))
    new_file_list = []
    for idx in sort_mask:
        new_file_list.append(file_list[idx])
    return new_file_list

# ========= Functions about the checkpoint
def save_checkpoint(args, model, which_part='alice', file_name='test'):
    if which_part.lower()=='alice':
        file_path = os.path.join(args.save_path, 'Alice_'+file_name+'.pth')
        if args.model.startswith('vit'):
            torch.save(model.state_dict(), file_path)
        elif args.model.startswith('res'):
            torch.save(model.Alice.state_dict(), file_path)
    elif which_part.lower()=='bob':
        file_path = os.path.join(args.save_path, 'Bob_' + file_name+'.pth')
        if args.model.startswith('vit'):
            torch.save(model.head.state_dict(), file_path)
        elif args.model.startswith('res'):
            torch.save(model.Bob.state_dict(), file_path)
    elif which_part.lower()=='all':
        file_path = os.path.join(args.save_path, 'All_'+file_name+'.pth')
        torch.save(model.state_dict(), file_path)
    else:
        print('which_part must be alice or bob or all')
        
def load_checkpoint(args, model, ckp_path, which_part='alice'):
    '''
        Use this to load params of specific part (Alice, Bob or all),
        from ckp to model.
    '''
    if which_part.lower()=='alice':
        if args.model.startswith('vit'):
            mis_k, unex_k = model.load_state_dict(torch.load(ckp_path),strict=False)
        elif args.model.startswith('res'):
            mis_k, unex_k = model.Alice.load_state_dict(torch.load(ckp_path),strict=False)
    elif which_part.lower()=='bob':
        if args.model.startswith('vit'):
            mis_k, unex_k = model.head.load_state_dict(torch.load(ckp_path),strict=False)
        elif args.model.startswith('res'):
            mis_k, unex_k = model.Bob.load_state_dict(torch.load(ckp_path),strict=False)
    elif which_part.lower()=='all':
        mis_k, unex_k = model.load_state_dict(torch.load(ckp_path),strict=False)    
    else:
        print('which_part must be alice or bob')
    return mis_k, unex_k


# =========== wandb functions =================
def wandb_init(proj_name='test', run_name=None, config_args=None):
    wandb.init(
        project=proj_name,
        config={}, entity="joshua_shawn",reinit=True)
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
        net = ResNet18(args.nb_classes, Bob_layer=args.Bob_layer, Bob_depth=args.Bob_depth)
    elif net_type=='resnet50':
        net = ResNet50(args.nb_classes, Bob_layer=args.Bob_layer)
    elif net_type=='vit':
        net = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=args.nb_classes)
    else:
        print('net structure not supported, only support resnet18, resnet50, vit16')
    return net

def get_optimizer(model, args):
    if args.optim_type.lower()=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=args.weight_decay,nesterov=True)
        if args.scheduler_type=='cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        elif args.scheduler_type=='multistep':
            s_ratio = [args.s1, args.s2, args.s3]
            #s_ratio = [100, 200, 400]
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=s_ratio,gamma=0.1)
    elif args.optim_type.lower()=='adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.scheduler_type=='cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        elif args.scheduler_type=='multistep':
            s_ratio = [args.s1, args.s2, args.s3]
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=s_ratio,gamma=0.1)
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