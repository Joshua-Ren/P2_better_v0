#---------------------------------------------------------
# Stage 1 is pre-train, i.e., generate checkpoint by training
# on a specific dataset.
# --------------------------------------------------------

import argparse
import numpy as np
import os
import time
from pathlib import Path
import models

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

from engine_PT import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('Stage1 generate pretrain checkpoint', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--pt_epochs', default=200, type=int)

    # Model parameters
    parser.add_argument('--model', default='resnet18', type=str, metavar='MODEL',
                        help='only for resnet 18')
    parser.add_argument('--figsize', default=32, type=int,
                        help='images input size, cifar is 32')
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
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='can be cifar10, cifar100, imagenet, domainnet')    
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')
                        
    parser.add_argument('--run_name',default=None,type=str)
    parser.add_argument('--proj_name',default='LP-FT-pretrain', type=str)
    
    parser.add_argument('--work_dir', default='./results/',
                        help='path of the pretrained checkpoint')
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
    # ================= Prepare for distributed training =====
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # ================== Prepare for the dataloader ===============
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    # =================== Initialize wandb ========================
    run_name = wandb_init(proj_name=args.proj_name, run_name=args.run_name, config_args=args)
    save_path = os.path.join(args.work_dir, run_name)
    args.save_path = save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    # ================== Create the model ==================
    model = get_init_net(args)
    model.to(args.device)

    optimizer, scheduler = get_optimizer(model, args)

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    if args.loss_type=='mse':
        criterion = torch.nn.MSELoss()
    print("criterion = %s" % str(criterion))

    for epoch in range(args.pt_epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, data_loader_train, optimizer, scheduler, epoch, mixup_fn, args=args)
        evaluate(data_loader_val, model, args.device, args)
    save_checkpoint(args, model, which_part='alice', file_name='resnet18_PT')  # Check whether OK to save the multiGPU model

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.dataset=='cifar10' or args.dataset=='stl10':
        args.nb_classes=10
    elif args.dataset=='cifar100':
        args.nb_classes=100
    main(args)
