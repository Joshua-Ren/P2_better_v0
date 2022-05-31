#---------------------------------------------------------
# Stage 2 is LP-FT, i.e., on a different dataset,
# linear probe and then finetune together
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import wandb
import torch

from timm.data import Mixup
from timm.utils import accuracy
from util.general import *
import util.misc as misc

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                    scheduler:torch.optim.lr_scheduler, epoch: int, 
                    mixup_fn: Optional[Mixup] = None, args=None, train_type='ft'):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train(True)

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if args.loss_type=='mse':
                y_oht = F.one_hot(targets, num_classes=args.nb_class).reshape(-1,1)
                loss = criterion(outputs.reshape(-1,1),y_oht.float())                
            else:
                loss = criterion(outputs, targets)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    
    # ----- At the end of epoch
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]
    prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
    losses.update(loss.data.item(), samples.size(0))
    top1.update(prec1.item(), samples.size(0))
    top5.update(prec5.item(), samples.size(0))   
    if misc.is_main_process():
        if train_type == 'ft':
            wandb.log({'ft_epoch':epoch})         
            wandb.log({'ft_learn_rate':lr})
            wandb.log({'ft_train_loss':losses.avg})
            wandb.log({'ft_train_top1':top1.avg})
            wandb.log({'ft_train_top5':top5.avg})
        elif train_type == 'lp':
            wandb.log({'lp_epoch':epoch})         
            wandb.log({'lp_learn_rate':lr})
            wandb.log({'lp_train_loss':losses.avg})
            wandb.log({'lp_train_top1':top1.avg})
            wandb.log({'lp_train_top5':top5.avg})
            
@torch.no_grad()
def evaluate(data_loader, model, device,args, train_type='ft'):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    for i, (images,targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(images)
            if args.loss_type=='mse':
                y_oht = F.one_hot(targets, num_classes=args.nb_class).reshape(-1,1)
                loss = criterion(outputs.reshape(-1,1),y_oht.float())                
            else:
                loss = criterion(outputs, targets)
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.data.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))
        
    if misc.is_main_process():
        if train_type == 'ft':
            wandb.log({'ft_valid_loss':losses.avg})
            wandb.log({'ft_valid_top1':top1.avg})
            wandb.log({'ft_valid_top5':top5.avg})
        elif train_type == 'lp':
            wandb.log({'lp_valid_loss':losses.avg})
            wandb.log({'lp_valid_top1':top1.avg})
            wandb.log({'lp_valid_top5':top5.avg})           
    return top1.avg, top5.avg