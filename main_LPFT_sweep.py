#---------------------------------------------------------
# Stage 2 is LP-FT, i.e., on a different dataset,
# linear probe and then finetune together
# --------------------------------------------------------

import argparse
import numpy as np
import os
import time
from pathlib import Path
import models
import copy

import torch
import torch.backends.cudnn as cudnn
import timm
#assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from util.general import *
from util.datasets import build_dataset
from engine_LPFT import train_one_epoch, evaluate, evaluate_ood

LP_EPOCHS = [0, 1, 2, 4, 8, 16, 32, 50, 100, 199]
LP_EPOCHS = LP_EPOCHS[::-1]

def get_args_parser():
    parser = argparse.ArgumentParser('Stage2 linear prob and finetune', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--FT_epochs', default=100, type=int)
    # Pretrain checkpoint
    parser.add_argument('--work_dir', default=None,
                        help='path of the pretrained checkpoint')
    parser.add_argument('--LP_dir', default=None, type=str,
                        help='Under work-dir, which LP dir we choose')
    parser.add_argument('--alice_name', default=None,
                        help='name of the pretrained checkpoint')
    parser.add_argument('--target_bob', default=None,
                        help='name of the target bob checkpoint')

    # Model parameters
    parser.add_argument('--model', default='resnet18', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--figsize', default=224, type=int,
                        help='images input size, all use 224')
    parser.add_argument('--Bob_layer', default=1, type=float,
                        help='1: only last fc, 1.3: fc+layer4(C), 1.6: fc+layer4(CB) 2: fc+layer4-all')
    parser.add_argument('--Bob_depth', default=1, type=int,
                        help='1: linear, 2: 2-layers MLP, 3: 3-layers MLP')
                        
    # Optimizer parameters
    parser.add_argument('--loss_type', type=str, default='ce',
                        help='can be mse or ce')    
    parser.add_argument('--optim_type', type=str, default='sgd',
                        help='can be sgd or adam')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    #parser.add_argument('--clip_grad', type=float, default=10, metavar='NORM',
    #                    help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup', type=int, default=0,
                        help='warmup FT_epochs')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                        help='can be cosine or multistep')
    parser.add_argument('--s1', type=int, default=100,
                        help='can be cosine or multistep')                            
    parser.add_argument('--s2', type=int, default=300,
                        help='can be cosine or multistep') 
    parser.add_argument('--s3', type=int, default=600,
                        help='can be cosine or multistep')        
                        
    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--smoothing', type=float, default=0,
                        help='Label smoothing (default: 0)')
    parser.add_argument('--FT_smoothing', type=float, default=0,
                        help='Label smoothing (default: 0)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--dataset', default='stl10', type=str,
                        help='can be cifar10, stl10, cifar100, imagenet, domain_quick')    
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')
                        
    parser.add_argument('--run_name',default=None,type=str)
    parser.add_argument('--proj_name',default='betterv0_FT', type=str)
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    if args.seed==0:
        args.seed = np.random.randint(1,10086)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    
    # ================== Prepare for the dataloader ===============
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    # =================== Initialize wandb ========================
    run_name = wandb_init(proj_name=args.proj_name, run_name=args.run_name, config_args=args)
    args.save_path = os.path.join(args.work_dir, run_name)
            # -------- save results in this folder
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # ================== Prepare for the dataloader ===============
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    # ================== Create the model and copy alice parameters ==================
    seed_model = get_init_net(args)
    alice_path = os.path.join(args.work_dir, args.alice_name)
    load_checkpoint(args, seed_model, alice_path, which_part='alice')
    # ================== Get some common settings ==================
#    if mixup_fn is not None:
#        # smoothing is handled with mixup label transform
#        criterion = SoftTargetCrossEntropy()
#    else:
#        criterion = torch.nn.CrossEntropyLoss()

    for gen in range(len(LP_EPOCHS)):
        bob_ep = LP_EPOCHS[gen]
    # ================== LP the network ============================
        if args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            if args.loss_type=='mse':
                criterion = torch.nn.MSELoss()
            elif args.loss_type=='ce':
                criterion = torch.nn.CrossEntropyLoss()
        tmp_warmup = copy.deepcopy(args.warmup)
        args.warmup = 0
        args.weight_decay = 0
        args.min_lr = args.lr
        model = copy.deepcopy(seed_model)
        model.to(args.device)
        if args.model in ['resnet18', 'resnet34', 'resnet50']:
            optim_bob, scheduler_bob = get_optimizer(model.Bob, args)
        elif args.model in ['vit16']:
            optim_bob, scheduler_bob = get_optimizer(model.head, args)
                # ---- Try new method: LS during LP, preserve energy.
        for epoch in range(bob_ep):
            evaluate(data_loader_val, model, args.device, args, train_type='lp')
            train_one_epoch(model, criterion, data_loader_train, optim_bob, scheduler_bob, epoch, mixup_fn, args=args, train_type='lp')
    # ================== FT the network ============================
        if args.FT_smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            if args.loss_type=='mse':
                criterion = torch.nn.MSELoss()
            elif args.loss_type=='ce':
                criterion = torch.nn.CrossEntropyLoss()
        args.warmup = tmp_warmup
        args.weight_decay = 5e-4
        #args.epochs = args.epochs*2
        args.lr = 5e-4
        args.min_lr = 1e-6#args.lr*0.02
        results = {'tloss':[],'tacc':[], #'tprobs':[],
                   'vloss':[],'vacc':[],'vprobs':[],
                'ztz0_cos':[], 'ztz0_norm':[],'ztz0_dot':[],'zt_norm':[], 
                'grad_bob':[]}
        #model0 = copy.deepcopy(seed_model)
        model0 = copy.deepcopy(model)
        model0.to(args.device)
        optimizer, scheduler = get_optimizer(model, args)
        best_vacc = 0
        for epoch in range(args.FT_epochs):
            vloss, vacc, vprobs, ztz0_cos, ztz0_norm, ztz0_dot, zt_norm = evaluate(data_loader_val, model, args.device, args, model0=model0, train_type='ft')         
            tloss, tacc, grad_bob = train_one_epoch(model, criterion, data_loader_train, optimizer, scheduler, epoch, mixup_fn, args=args, train_type='ft')  
            if vacc >= best_vacc:
                best_vacc = vacc
            results['tloss'].append(tloss)
            results['tacc'].append(tacc)
            results['vloss'].append(vloss)
            results['vacc'].append(vacc)
            results['vprobs'].append(vprobs)
            results['ztz0_cos'].append(ztz0_cos)
            results['ztz0_norm'].append(ztz0_norm)
            results['ztz0_dot'].append(ztz0_dot)
            results['zt_norm'].append(zt_norm)
            results['grad_bob'].append(grad_bob)
        wandb.log({'ft_last':vacc})
        wandb.log({'ft_best':best_vacc})
        wandb.log({'ft_bob_ep':bob_ep})
        # ----- Save the npy
        result_save_name = os.path.join(args.save_path, 'Bob_ep'+str(bob_ep).zfill(4)+'.npy')
        np.save(result_save_name, results)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args = args_get_class(args)
    main(args)