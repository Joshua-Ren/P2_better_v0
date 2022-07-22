# -*- coding: utf-8 -*-
"""
For resnet18 and other 4.1 experiments, always keep AB_split = 6 (clone all backbone)
For experiments in 4.2, can set AB_split to 1,2,3
"""

import argparse
import numpy as np
import os
import time
from pathlib import Path
import models
import copy
import torch
import torch.backends.cudnn as cudnn
from util.general import *
from torchvision import models as M


RES_CKP = ['Alice_resnet18_PT.pth', 'All_resnet18_PT.pth']
ckp_folder = 'E:\\P2_better_v0\\analysis\\download_checkpoints'

def get_args_parser():
    parser = argparse.ArgumentParser('Stage2 linear prob one GPU', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--figsize', default=224, type=int,
                        help='images input size, cifar is 32')
    parser.add_argument('--Bob_layer', default=1, type=float,
                        help='1: only last fc, 1.3: fc+layer4(C), 1.6: fc+layer4(CB) 2: fc+layer4-all')
    parser.add_argument('--Bob_depth', default=1, type=int,
                        help='1: linear, 2: 2-layers MLP, 3: 3-layers MLP')    
    parser.add_argument('--nb_classes', default=10)
    return parser

args = get_args_parser()
args = args.parse_args()

ckp_path = os.path.join(ckp_folder,'resnet50-simclr.pth')
ckp_dict = torch.load(ckp_path)
ckp_key_list = list(ckp_dict.keys())

seed_model = get_init_net(args)
seed_model = seed_model.cuda()
seed_dict = seed_model.state_dict()
seed_key_list = list(seed_dict.keys())

load_model = copy.deepcopy(seed_model)
missing_keys, unexpected_keys = load_model.Alice.load_state_dict(ckp_dict, strict=False)

torch.save(load_model.Bob.state_dict(), 'test.bob')
mis_k, unex_k = load_model.Bob.load_state_dict(torch.load('test.bob'),strict=False)
"""
n_list = []
for n,p in load_model.named_parameters():
    n_list.append(n)

z,h = seed_model(torch.randn(4,3,224,224).cuda())
zdot = torch.bmm(z.unsqueeze(1),z.unsqueeze(2)).detach().mean()

"""
# -------- How to load the Alice's parameters
    # Alice's parameters comes from downloaded checkpoint (or saved during stage1)


# ------ In stage1, we should save the model using the following way:
#torch.save(seed_model.Alice.state_dict(),ckp_folder+'/Alice.pth')

#Alice_ckp_path = os.path.join(ckp_folder,'Alice.pth')
#Alice_ckp_dict = torch.load(Alice_ckp_path)
#Alice_ckp_key_list = list(Alice_ckp_dict.keys())

#missing_keys, unexpected_keys = seed_model.Alice.load_state_dict(Alice_ckp_dict, strict=False)

