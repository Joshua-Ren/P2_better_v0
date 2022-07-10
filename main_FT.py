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
from engine_LPFT import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('Stage2 linear prob and finetune', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=100, type=int)

    # Pretrain checkpoint
    parser.add_argument('--work_dir', default=None,
                        help='path of the pretrained checkpoint')
    parser.add_argument('--LP_dir', default=None, type=str,
                        help='Under work-dir, which LP dir we choose')
    parser.add_argument('--alice_name', default=None,
                        help='name of the pretrained checkpoint')


    # Model parameters
    parser.add_argument('--model', default='resnet18', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--figsize', default=224, type=int,
                        help='images input size, all use 224')
    parser.add_argument('--Bob_layer', default=1, type=int,
                        help='1: only last fc, 2: fc+layer4, 3:fc+layer43, 4: fc+layer432')

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
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--smoothing', type=float, default=0,
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

     # ------- For domainnet, we have OOD vacc
    if args.dataset=='domain_quick':
        dataset_val_ood1 = build_dataset(is_train=False, args=args,force_dataset='domain_sketch')
        dataset_val_ood2 = build_dataset(is_train=False, args=args,force_dataset='domain_real')
    elif args.dataset=='domain_sketch':
        dataset_val_ood1 = build_dataset(is_train=False, args=args,force_dataset='domain_quick')
        dataset_val_ood2 = build_dataset(is_train=False, args=args,force_dataset='domain_real')
    elif args.dataset=='domain_real':
        dataset_val_ood1 = build_dataset(is_train=False, args=args,force_dataset='domain_quick')
        dataset_val_ood2 = build_dataset(is_train=False, args=args,force_dataset='domain_sketch')        
    
    if args.dataset[:6]=='domain':
        data_loader_val_ood1 = torch.utils.data.DataLoader(dataset_val_ood1,
                                batch_size=args.batch_size, num_workers=args.num_workers,
                                pin_memory=args.pin_mem, drop_last=True)
        data_loader_val_ood2 = torch.utils.data.DataLoader(dataset_val_ood2,
                                batch_size=args.batch_size, num_workers=args.num_workers,
                                pin_memory=args.pin_mem, drop_last=True)
    
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
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    if args.loss_type=='mse':
        criterion = torch.nn.MSELoss()

    # ================== FT all parts, use multiple GPUs
        # ----- Get all checkpoints for Bob
    bob_ckp_folder = os.path.join(args.work_dir, args.LP_dir)
    file_list = sort_files(os.listdir(bob_ckp_folder))
        # ----- Search the file, FT all the Bob checkpoints
    for i in range(len(file_list)):
        results = {'tloss':[],'tacc':[], #'tprobs':[],
                   'vloss':[],'vacc':[],'vprobs':[], 'vacc_o1':[], 'vacc_o2':[], 
                'ztz0_cos':[], 'ztz0_norm':[],'ztz0_dot':[],'zt_norm':[], 
                'grad_bob':[]}
        modelt = copy.deepcopy(seed_model)
        model0 = copy.deepcopy(modelt)
        modelt.to(args.device)
        model0.to(args.device)

        f = file_list[i]
        bob_ep = int(f.split('.')[0].split('_')[-1])
        bob_path = os.path.join(bob_ckp_folder, f)
        load_checkpoint(args, modelt, bob_path, which_part='bob')
        optimizer, scheduler = get_optimizer(modelt, args)
        best_vacc = 0
        for epoch in range(args.epochs):
            vloss, vacc, vprobs, ztz0_cos, ztz0_norm, ztz0_dot, zt_norm = evaluate(data_loader_val, modelt, args.device, args, model0=model0, train_type='ft')
            if args.dataset[:6]=='domain':
                vacc_o1 = evaluate_ood(data_loader_val_ood1, modelt, args.device, args,wb_title='ft_valid_ood1')
                vacc_o2 = evaluate_ood(data_loader_val_ood2, modelt, args.device, args,wb_title='ft_valid_ood2')
                results['vacc_o1'].append(vacc_o1)
                results['vacc_o2'].append(vacc_o2)            
            tloss, tacc, grad_bob = train_one_epoch(modelt, criterion, data_loader_train, optimizer, scheduler, epoch, mixup_fn, args=args, train_type='ft')  
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
        result_save_name = os.path.join(args.save_path, f[:-3]+'npy')
        np.save(result_save_name, results)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args = args_get_class(args)
    main(args)