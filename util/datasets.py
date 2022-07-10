# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

import torchvision
import torch
from torchvision import datasets
import torchvision.transforms as T
import torch.utils.data as Data
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .data_loader_lmdb import ImageFolderLMDB
import numpy as np

DATA_PATH = '/home/sg955/rds/rds-nlp-cdt-VR7brx3H4V8/datasets/'
def build_dataset(is_train, args, force_dataset=None):
    if force_dataset is None:
        dataset_name = args.dataset
    else:
        dataset_name = force_dataset

    # --------- For small datasets, get transform
    train_T, val_T = get_std_transform(figsize=args.figsize)
    
    if dataset_name=='imagenet':
        transform = build_imagenet_transform(is_train, args)
        args.data_path = DATA_PATH+'ImageNet/'
        root = os.path.join(args.data_path, 'train.lmdb' if is_train else 'val.lmdb')
        dataset = ImageFolderLMDB(root, transform=transform)
    elif dataset_name=='stl10':
        if is_train:
            dataset = torchvision.datasets.STL10(DATA_PATH+'stl10', split='train', download=True, transform=train_T)
        else:
            dataset = torchvision.datasets.STL10(DATA_PATH+'stl10', split='test', download=True, transform=val_T)
    elif dataset_name=='cifar10':
        if is_train:
            dataset = torchvision.datasets.CIFAR10(DATA_PATH+'cifar10', train=True, download=True, transform=train_T)
        else:
            dataset = torchvision.datasets.CIFAR10(DATA_PATH+'cifar10', train=False, download=True, transform=val_T)
    elif dataset_name=='cifar10p':
        mean, std =  np.array([0.4914, 0.4822, 0.4465]), np.array([0.2023, 0.1994, 0.2010])
        # ------ This dataset only supports figsize=32, and no augmentation
        if is_train:
            x, y = np.load(DATA_PATH+'cifar10p1/cifar10.1_v4_data.npy'), np.load(DATA_PATH+'cifar10p1/cifar10.1_v4_labels.npy')
            x = (x/256-mean)/std
            dataset = Data.TensorDataset(torch.tensor(x), torch.tensor(y))
        else:
            x, y = np.load(DATA_PATH+'cifar10p1/cifar10.1_v6_data.npy'), np.load(DATA_PATH+'cifar10p1/cifar10.1_v6_labels.npy')
            x = (x/256-mean)/std
            dataset = Data.TensorDataset(torch.tensor(x), torch.tensor(y))
    elif dataset_name=='cifar100':
        if is_train:
            dataset = torchvision.datasets.CIFAR100(DATA_PATH+'cifar100', train=True, download=True, transform=train_T)
            #dataset = Data.TensorDataset(torch.tensor(origin_dataset.data[:10000]).transpose(1,3), torch.tensor(origin_dataset.targets[:10000]))
        else:
            dataset = torchvision.datasets.CIFAR100(DATA_PATH+'cifar100', train=False, download=True, transform=val_T)
    elif dataset_name == 'domain_quick':
        if is_train:
            dataset = torchvision.datasets.ImageFolder(DATA_PATH+'domain/quick/train', transform=train_T)
        else:
            dataset = torchvision.datasets.ImageFolder(DATA_PATH+'domain/quick/val', transform=val_T)
    elif dataset_name == 'domain_sketch':
        if is_train:
            dataset = torchvision.datasets.ImageFolder(DATA_PATH+'domain/sketch/train', transform=train_T)
        else:
            dataset = torchvision.datasets.ImageFolder(DATA_PATH+'domain/sketch/val', transform=val_T)
    elif dataset_name == 'domain_real':
        transform = build_imagenet_transform(is_train, args)
        args.data_path = DATA_PATH+'domain/real/'
        root = os.path.join(args.data_path, 'train.lmdb' if is_train else 'val.lmdb')
        dataset = ImageFolderLMDB(root, transform=transform)       
    return dataset

def get_std_transform(figsize=32):
    """
        For CIFAR10/100, STL, Domain Net or other small dataset, use this
    """
    train_T=T.Compose([
                    T.Resize([figsize,figsize]),
                    T.RandomCrop(figsize, padding=int(figsize*0.2)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
    val_T =T.Compose([
                    T.Resize([figsize,figsize]),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
    return train_T, val_T
    

def build_imagenet_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        T.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(T.CenterCrop(args.input_size))

    t.append(T.ToTensor())
    t.append(T.Normalize(mean, std))
    return T.Compose(t)
