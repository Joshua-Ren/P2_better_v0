'''
    Change the ResNet function to make Alice/Bob split, out put change to z, hid
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def ResNet34(num_classes, AB_split):
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes=num_classes, AB_split=AB_split)


def ResNet50(num_classes, AB_split):
    return ResNet(Bottleneck, [3, 4, 6, 3],num_classes=num_classes, AB_split=AB_split)


def ResNet101(num_classes, AB_split):
    return ResNet(Bottleneck, [3, 4, 23, 3],num_classes=num_classes, AB_split=AB_split)


def ResNet152(num_classes, AB_split):
    return ResNet(Bottleneck, [3, 8, 36, 3],num_classes=num_classes, AB_split=AB_split)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
