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
#import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
#from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_LPFT import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('Stage2 linear prob one GPU', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=8192, type=int)
    parser.add_argument('--scheduler_epochs', default=8192, type=int)
    parser.add_argument('--lp_epoch_list',default=[0, 1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], type=list,
                        help='which vector_ep we select for the FT phase')

    # Pretrain checkpoint
    parser.add_argument('--work_dir', default=None,
                        help='path of the pretrained checkpoint')
    parser.add_argument('--alice_name', default=None,
                        help='name of the pretrained checkpoint')
                        
    # Model parameters
    parser.add_argument('--model', default='resnet18', type=str, metavar='MODEL',
                        help='Name of model to train, resnet18, resnet50, vit16')
    parser.add_argument('--figsize', default=224, type=int,
                        help='images input size, all use 224')
    parser.add_argument('--Bob_layer', default=1, type=int,
                        help='1: only last fc, 2: fc+layer4, 3:fc+layer43, 4: fc+layer432')

    # Optimizer parameters
    parser.add_argument('--loss_type', type=str, default='ce',
                        help='can be mse or ce')    
    parser.add_argument('--optim_type', type=str, default='sgd',
                        help='can be sgd or adam')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--min_lr', type=float, default=1e-4, metavar='LR',
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
                        help='can be cifar10.1, stl10, cifar100, tiny, domainnet')    
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')
                        
    parser.add_argument('--run_name',default=None,type=str)
    parser.add_argument('--proj_name',default='betterv0_LP', type=str)
    
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
    if args.seed==-1:
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
            # -------- save bob's checkpoints in this folder
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
    mis_k, unex_k = load_checkpoint(args, seed_model, alice_path, which_part='all')
    #mis_k, unex_k = load_checkpoint(args, seed_model, alice_path, which_part='alice')
    print('=======mis_k============')
    print(mis_k)
    print('=======unex_k============')
    print(unex_k)
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

    # ================== LP Bob part, save dict for args.lp_epoch_list
    model = copy.deepcopy(seed_model)
    model.to(args.device)
    if args.model in ['resnet18', 'resnet50']:
        optim_bob, scheduler_bob = get_optimizer(model.Bob, args)
    elif args.model in ['vit16']:
        optim_bob, scheduler_bob = get_optimizer(model.head, args)
    for epoch in range(args.epochs):
        if epoch in args.lp_epoch_list:
            ckp_name = 'ep_'+str(epoch).zfill(4)
            save_checkpoint(args, model, which_part='bob', file_name=ckp_name)
        train_one_epoch(model, criterion, data_loader_train, optim_bob, scheduler_bob, epoch, mixup_fn, args=args, train_type='lp')
        evaluate(data_loader_val, model, args.device, args, train_type='lp')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args = args_get_class(args)
    main(args)