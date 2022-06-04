import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import copy
import matplotlib.tri as tri
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torch.autograd import Variable
import torch.utils.data as Data 
from torch.utils.data.sampler import SubsetRandomSampler






def get_Alice_Bob_dict(tmp_model):
    from collections import OrderedDict
    alice_dict = OrderedDict()
    bob_dict = OrderedDict()
    for name, param in tmp_model.Alice.named_parameters():
        alice_dict[name] = param
    for name, param in tmp_model.Bob.named_parameters():
        bob_dict[name] = param
    return alice_dict, bob_dict

def load_checkpoint(model, ckp_path, which_part='all'):
    '''
        Use this to load params of specific part (Alice, Bob or all),
        from ckp to model.
    '''
    if which_part.lower()=='all':
        model.load_state_dict(torch.load(ckp_path))
    elif which_part.lower()=='alice':
        tmp_model = ResNet18(10, 6)
        tmp_model.load_state_dict(torch.load(ckp_path))
        alice_dict, _ = get_Alice_Bob_dict(tmp_model)
        model.Alice.load_state_dict(alice_dict,strict=False)
    elif which_part.lower()=='bob':
        tmp_model = ResNet18(10, 6)
        tmp_model.load_state_dict(torch.load(ckp_path))
        _, bob_dict = get_Alice_Bob_dict(tmp_model)
        model.Bob.load_state_dict(bob_dict,strict=False)
    else:
        print('which_part must be alice, bob, or all')    

def cal_ECE(pb_table, tf_table):
  '''
    pb_table is the probability provided by network
    tf_table is the acc results of the prodiction
  '''
  BM_acc = np.zeros((10,))
  BM_conf = np.zeros((10,))
  BM_cnt = np.zeros((10,))
  Index_table = ((pb_table-1e-6).T*10).int().squeeze()

  for i in range(pb_table.shape[0]):
    idx = Index_table[i]
    BM_cnt[idx] += 1
    BM_conf[idx] += pb_table[i]
    if tf_table[i]:
      BM_acc[idx] += 1
  ECE = 0
  for j in range(10):
    if BM_cnt[j] != 0:
      ECE += BM_cnt[j]*np.abs(BM_acc[j]/BM_cnt[j]-BM_conf[j]/BM_cnt[j])
  return ECE/BM_cnt.sum()

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, AB_split=6):
      # Alice_Bob_split should be 1, 2, 3, 4, or 6, meaning split after this number
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer0 = nn.Sequential(self.conv1, self.bn1, nn.ReLU())
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.pool2d = nn.AvgPool2d(kernel_size=4)
        self.view = nn.Flatten()
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.AB_split = AB_split
        self.Alice, self.Bob = self._Alice_Bob_split()

    def _Alice_Bob_split(self):
      layer_list = [self.layer0, self.layer1, self.layer2, self.layer3, 
              self.layer4, self.pool2d, self.view, self.linear]
      name_list = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'pool2d', 'view', 'linear']
      Alice, Bob = nn.Sequential(), nn.Sequential()
      for i in range(len(name_list)):
        if i<= self.AB_split:
          Alice.add_module(name_list[i],layer_list[i])
          print('Alice contains '+name_list[i])
        else:
          Bob.add_module(name_list[i],layer_list[i])
          print('Bob contains '+name_list[i])
      return Alice,Bob

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.Alice(x)
        hid = self.Bob(z)
        return z, hid


def ResNet18(num_classes, AB_split):
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes=num_classes, AB_split=AB_split)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res