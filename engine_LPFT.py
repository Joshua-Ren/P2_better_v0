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
    grad_bob = AverageMeter()
    model.train(True)
    for data_iter_step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.cuda.amp.autocast():
            _, outputs = model(samples)
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
        tmp_grad_bob = get_bob_grad_norm(args, model)
        grad_bob.update(tmp_grad_bob)
        optimizer.step()
        
    # ----- At the end of epoch
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]
    prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
    losses.update(loss.data.item(), samples.size(0))
    top1.update(prec1.item(), samples.size(0))
    if misc.is_main_process():
        if train_type == 'ft':
            wandb.log({'ft_epoch':epoch})         
            wandb.log({'ft_learn_rate':lr})
            wandb.log({'ft_train_loss':losses.avg})
            wandb.log({'ft_train_top1':top1.avg})
            wandb.log({'ft_train_bobnrom':grad_bob.avg})
        elif train_type == 'lp':
            wandb.log({'lp_epoch':epoch})         
            wandb.log({'lp_learn_rate':lr})
            wandb.log({'lp_train_loss':losses.avg})
            wandb.log({'lp_train_top1':top1.avg})
    return losses.avg, top1.avg, grad_bob.avg  
    
@torch.no_grad()
def evaluate(data_loader, model, device, args, model0=None, train_type='ft'):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    ztz0_cos = AverageMeter()
    ztz0_norm = AverageMeter()
    ztz0_dot = AverageMeter()
    zt_norm = AverageMeter()    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    if model0 is not None:
        model0.eval()
    pb_table = []
    for i, (images,targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            zt, hid = model(images)
            if model0 is not None:
                # ---- may need some change for vit model
                z0, _ = model0(images)
                cos_dist = torch.nn.CosineSimilarity(dim=1)(z0,zt).detach().mean()
                norm_dist = torch.norm(zt-z0, dim=1).detach().mean()
                dot_dist = torch.bmm(zt.unsqueeze(1),z0.unsqueeze(2)).detach().mean()
                print(zt)
                print(z0)
                zt_dist = torch.norm(zt,dim=1).detach().mean()
                ztz0_cos.update(cos_dist,targets.size(0))
                ztz0_norm.update(norm_dist,targets.size(0))
                if abs(dot_dist)<10000:
                    ztz0_dot.update(dot_dist.cpu(),targets.size(0))
                zt_norm.update(zt_dist.cpu(),targets.size(0))

            hid = hid.detach()
            pred_idx = hid.data.max(1, keepdim=True)[1]
            if args.loss_type=='mse':
                prob = torch.gather(hid,dim=1, index=pred_idx)
                y_oht = F.one_hot(targets, num_classes=args.nb_class).reshape(-1,1)
                loss = criterion(hid.reshape(-1,1),y_oht.float())                
            else:
                prob = torch.gather(nn.Softmax(1)(hid),dim=1, index=pred_idx)
                loss = criterion(hid, targets)
            pb_table.append(prob.cpu())
        prec1, prec5 = accuracy(hid, targets, topk=(1, 5))
        losses.update(loss.data.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
    pb_table = np.array(torch.stack(pb_table).reshape(-1,1))
    
    if train_type == 'ft':
        wandb.log({'ft_valid_loss':losses.avg})
        wandb.log({'ft_valid_top1':top1.avg})
        if model0 is not None:
            wandb.log({'ztz0_cos':ztz0_cos.avg})
            wandb.log({'ztz0_norm':ztz0_norm.avg})        
            wandb.log({'ztz0_dot':ztz0_dot.avg})
            wandb.log({'zt_norm':zt_norm.avg})              
    elif train_type == 'lp':
        wandb.log({'lp_valid_loss':losses.avg})
        wandb.log({'lp_valid_top1':top1.avg})        
    return losses.avg, top1.avg, pb_table, ztz0_cos.avg, ztz0_norm.avg, ztz0_dot.avg, zt_norm.avg