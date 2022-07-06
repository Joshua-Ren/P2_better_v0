# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:41:26 2022

@author: YIREN
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
import torchvision.transforms as T
import torch.utils.data as Data

# ------------ CIFAR10.1
DATA_FOLDER = 'E:\\P2_better_v0\\data\\cifar10_1'
datav4 = np.load(os.path.join(DATA_FOLDER,'cifar10.1_v4_data.npy'))
labelv4 = np.load(os.path.join(DATA_FOLDER,'cifar10.1_v4_labels.npy'))
datav6 = np.load(os.path.join(DATA_FOLDER,'cifar10.1_v6_data.npy'))
labelv6 = np.load(os.path.join(DATA_FOLDER,'cifar10.1_v6_labels.npy'))
data_train = Data.TensorDataset(torch.tensor(datav4/256), torch.tensor(labelv4))
data_test = Data.TensorDataset(torch.tensor(datav6/256), torch.tensor(labelv6))
