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
import timm

VIT_CKP = ['vitbase-mae.pth', 'vitbase-dino.pth', 'vitbase-classification.pth']
ckp_folder = os.path.join('E:\\P2_better_v0\\','analysis','download_checkpoints')

def get_args_parser():
    parser = argparse.ArgumentParser('Stage2 linear prob one GPU', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='vit', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--figsize', default=32, type=int,
                        help='images input size, cifar is 32')
    parser.add_argument('--Bob_layer', default=1, type=int,
                        help='1: only last fc, 2: fc+layer4, 3:fc+layer43, 4: fc+layer432')
    parser.add_argument('--nb_classes', default=1000)
    return parser

args = get_args_parser()
args = args.parse_args()

dino_path = os.path.join(ckp_folder,'vitbase-dino.pth')
dino_dict = torch.load(dino_path)
dino_key_list = list(dino_dict.keys())

clas_path = os.path.join(ckp_folder,'vitbase-classification.pth')
clas_dict = torch.load(clas_path)
clas_key_list = list(clas_dict.keys())

vitbase = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10)


vitbase_dict = vitbase.state_dict()
vitbase_key_list = list(vitbase_dict.keys())


vitbase.load_state_dict(clas_dict, strict=False)

'''
seed_model = get_init_net(args)
seed_dict = seed_model.state_dict()
seed_key_list = list(seed_dict.keys())

# -------- How to load the Alice's parameters
    # Alice's parameters comes from downloaded checkpoint (or saved during stage1)
missing_keys, unexpected_keys = seed_model.Alice.load_state_dict(ckp_dict, strict=False)

# ------ In stage1, we should save the model using the following way:
torch.save(seed_model.Alice.state_dict(),ckp_folder+'/Alice.pth')
torch.save(seed_model.Bob.state_dict())

Alice_ckp_path = os.path.join(ckp_folder,'Alice.pth')
Alice_ckp_dict = torch.load(Alice_ckp_path)
Alice_ckp_key_list = list(Alice_ckp_dict.keys())

missing_keys, unexpected_keys = seed_model.Alice.load_state_dict(Alice_ckp_dict, strict=False)
'''
