# Copyright (C) 2024 Björn A. Lindqvist
#
# DenseNet from scratch and trained on CIFAR10.
#
# Code from: https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py
from itertools import islice
from math import floor
from pathlib import Path
from torch.nn.functional import avg_pool2d, cross_entropy, mse_loss, one_hot, relu
from torch.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    Parameter,
    ReLU,
    Sequential,
)
from torch.nn.init import constant_, kaiming_normal_, normal_
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchinfo import summary
from torchtoolbox.transform import Cutout
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

import numpy as np
import random
import torch

N_CLS = 10
BS = 128
DATA_DIR = Path("/tmp/data")
LR = 0.1
N_EPOCHS = 500
T_MAX = 50
SGD_MOM = 0.9

GROWTH_RATE = 32
N_BLOCKS = [6, 12, 24, 16]
REDUCTION = 0.5

class Bottleneck(Module):
    def __init__(self, n_chan_in):
        super(Bottleneck, self).__init__()
        self.bn1 = BatchNorm2d(n_chan_in)
        self.conv1 = Conv2d(n_chan_in, 4*GROWTH_RATE, 1, bias=False)
        self.bn2 = BatchNorm2d(4*GROWTH_RATE)
        self.conv2 = Conv2d(4*GROWTH_RATE, GROWTH_RATE, 3, padding=1, bias=False)

    def forward(self, x):
        xp = self.conv1(relu(self.bn1(x)))
        xp = self.conv2(relu(self.bn2(xp)))
        return torch.cat([xp,x], 1)

def build_transition(n_chans_in, n_chans_out):
    return Sequential(
        BatchNorm2d(n_chans_in),
        ReLU(),
        Conv2d(n_chans_in, n_chans_out, 1, bias=False),
        AvgPool2d(2)
    )

def build_dense_layers(n_chans, n_block):
    layers = []
    for i in range(n_block):
        layers.append(Bottleneck(n_chans))
        n_chans += GROWTH_RATE
    return Sequential(*layers)

# Performance of some variants:
#
# |VERSION               |ACC |
# +----------------------+----+
# |Standard              |94.4|
class DenseNet(Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        n_chans = 2 * GROWTH_RATE
        self.conv1 = Conv2d(3, n_chans, 3, padding=1, bias=False)

        self.dense1 = build_dense_layers(n_chans, N_BLOCKS[0])
        n_chans += N_BLOCKS[0] * GROWTH_RATE
        n_chans_out = int(floor(n_chans*REDUCTION))
        self.trans1 = build_transition(n_chans, n_chans_out)
        n_chans = n_chans_out

        self.dense2 = build_dense_layers(n_chans, N_BLOCKS[1])
        n_chans += N_BLOCKS[1] * GROWTH_RATE
        n_chans_out = int(floor(n_chans*REDUCTION))
        self.trans2 = build_transition(n_chans, n_chans_out)
        n_chans = n_chans_out

        self.dense3 = build_dense_layers(n_chans, N_BLOCKS[2])
        n_chans += N_BLOCKS[2] * GROWTH_RATE
        n_chans_out = int(floor(n_chans*REDUCTION))
        self.trans3 = build_transition(n_chans, n_chans_out)
        n_chans = n_chans_out

        self.dense4 = build_dense_layers(n_chans, N_BLOCKS[3])
        n_chans += N_BLOCKS[3] * GROWTH_RATE

        self.linear = Sequential(
            BatchNorm2d(n_chans),
            ReLU(),
            AvgPool2d(4),
            Flatten(),
            Linear(n_chans, N_CLS)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)
        x = self.linear(x)
        return x

def propagate_epoch(net, opt, loader, epoch):
    phase = "train" if net.training else "test"
    args = phase, epoch, N_EPOCHS
    print("== %s %3d/%3d ==" % args)
    tot_loss = 0
    tot_acc = 0
    n = len(loader)
    for i, (x, y) in enumerate(islice(loader, n)):
        if net.training:
            opt.zero_grad()
        yh = net(x)
        loss = cross_entropy(yh, y)
        if net.training:
            loss.backward()
            opt.step()
        loss = loss.item()
        acc = (yh.argmax(1) == y).sum().item() / BS
        print("%4d/%4d, loss/acc: %.4f/%.2f" % (i, n, loss, acc))
        tot_loss += loss
        tot_acc += acc
    return tot_loss / n, tot_acc / n

def main():
    norm = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trans_tr = Compose([
        RandomCrop(32, padding=4),
        Cutout(),
        RandomHorizontalFlip(),
        ToTensor(),
        norm
    ])
    trans_te = Compose([ToTensor(), norm])
    d_tr = CIFAR10(DATA_DIR, True, trans_tr, download = True)
    d_te = CIFAR10(DATA_DIR, False, trans_te, download = True)
    l_tr = DataLoader(d_tr, batch_size=BS, shuffle=False, drop_last=True)
    l_te = DataLoader(d_te, batch_size=BS, shuffle=True, drop_last=True)

    net = DenseNet()
    summary(net, input_size=(1, 3, 32, 32), device="cpu")
    opt = SGD(net.parameters(), LR, SGD_MOM)
    sched = CosineAnnealingLR(opt, T_max=T_MAX)
    max_te_acc = 0
    for i in range(N_EPOCHS):
        net.train()
        tr_loss, tr_acc = propagate_epoch(net, opt, l_tr, i)
        net.eval()
        with torch.no_grad():
            te_loss, te_acc = propagate_epoch(net, opt, l_te, i)
        max_te_acc = max(max_te_acc, te_acc)
        fmt = "losses %5.3f/%5.3f acc %5.3f/%5.3f, best acc %5.3f"
        print(fmt % (tr_loss, te_loss, tr_acc, te_acc, max_te_acc))
        sched.step()
main()
