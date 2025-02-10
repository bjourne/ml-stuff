# Copyright (C) 2024-2025 Björn A. Lindqvist <bjourne@gmail.com>
#
# DenseNet from scratch and trained on CIFAR10.
#
# Code from: https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py
from itertools import islice
from math import floor
from os import environ
from pathlib import Path
from time import time
from torch.distributed import (
    barrier, destroy_process_group, init_process_group
)
from torch.nn.functional import cross_entropy, relu
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
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
import torch.multiprocessing as mp

N_CLS = 10
BS = 256
DATA_DIR = Path("/tmp/data")
LR = 0.1
N_EPOCHS = 500
T_MAX = 50
SGD_MOM = 0.9

PRINT_INT = 1

GL_RANK = environ.get("RANK")
LO_RANK = environ.get("LOCAL_RANK")
IS_DISTR = LO_RANK is not None

DEV = LO_RANK
if IS_DISTR:
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################
# Model definition
########################################################################
GROWTH_RATE = 32
N_BLOCKS = [6, 12, 24, 16]
REDUCTION = 0.5


class Bottleneck(Module):
    def __init__(self, n_in):
        super(Bottleneck, self).__init__()
        self.bn1 = BatchNorm2d(n_in)
        self.conv1 = Conv2d(n_in, 4 * GROWTH_RATE, 1, bias=False)
        self.bn2 = BatchNorm2d(4 * GROWTH_RATE)
        self.conv2 = Conv2d(
            4 * GROWTH_RATE,
            GROWTH_RATE,
            3,
            padding=1,
            bias=False
        )

    def forward(self, x):
        xp = self.conv1(relu(self.bn1(x)))
        xp = self.conv2(relu(self.bn2(xp)))
        return torch.cat([xp, x], 1)

def build_transition(n_in, n_out):
    return Sequential(
        BatchNorm2d(n_in),
        ReLU(),
        Conv2d(n_in, n_out, 1, bias=False),
        AvgPool2d(2)
    )

def build_dense_layers(n_chans, n_block):
    layers = []
    for i in range(n_block):
        layers.append(Bottleneck(n_chans))
        n_chans += GROWTH_RATE
    return Sequential(*layers)

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

########################################################################
# Data utils
########################################################################
def load_cifar10(data_dir, bs, distributed):
    norm = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trans_tr = Compose([
        RandomCrop(32, padding=4),
        Cutout(),
        RandomHorizontalFlip(),
        ToTensor(),
        norm
    ])
    trans_te = Compose([ToTensor(), norm])
    d_tr = CIFAR10(data_dir, True, trans_tr, download = True)
    d_te = CIFAR10(data_dir, False, trans_te, download = True)
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset=d_tr)
    l_tr = DataLoader(
        d_tr,
        batch_size=bs,
        shuffle=False,
        drop_last=True,
        sampler=sampler,
        num_workers=16
    )
    l_te = DataLoader(
        d_te,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=16
    )
    return l_tr, l_te

########################################################################
# Training
########################################################################
def evaluate(net, l_te, epoch, tr_loss, tr_acc, tr_dur, max_te_acc):
    net.eval()
    with torch.no_grad():
        te_loss, te_acc, te_dur = propagate_epoch(net, None, l_te, epoch)
    max_te_acc = max(max_te_acc, te_acc)
    fmt = (
        "losses %5.3f/%5.3f "
        "acc %5.3f/%5.3f, "
        "best acc %5.3f, "
        "dur %5.2f/%5.2f"
    )
    args = (
        tr_loss, te_loss,
        tr_acc, te_acc,
        max_te_acc,
        tr_dur, te_dur
    )
    if not LO_RANK:
        print(fmt % args)
    return max_te_acc


def propagate_epoch(net, opt, loader, epoch):
    bef = time()
    print(LO_RANK)
    if not LO_RANK:
        phase = "train" if net.training else "test"
        args = phase, epoch, N_EPOCHS
        print("== %s %3d/%3d ==" % args)
    tot_loss = 0
    tot_acc = 0
    n = len(loader)
    for i, (x, y) in enumerate(islice(loader, n)):
        if net.training:
            opt.zero_grad()
        x = x.to(DEV)
        y = y.to(DEV)
        yh = net(x)
        loss = cross_entropy(yh, y)
        if net.training:
            loss.backward()
            opt.step()
        loss = loss.item()
        acc = (yh.argmax(1) == y).sum().item() / BS
        if i % PRINT_INT == 0 and not LO_RANK:
            print("%4d/%4d, loss/acc: %.4f/%.2f" % (i, n, loss, acc))
        tot_loss += loss
        tot_acc += acc
    return tot_loss / n, tot_acc / n, time() - bef

def main():
    n_devs = torch.cuda.device_count()
    if IS_DISTR:
        init_process_group(backend="nccl")
        torch.cuda.set_device(LO_RANK)
    print("Process %d/%d" % (LO_RANK or 0, n_devs))

    l_tr, l_te = load_cifar10(DATA_DIR, BS, IS_DISTR)

    net = DenseNet()
    net = net.to(DEV)
    if IS_DISTR:
        net = DistributedDataParallel(
            net,
            device_ids = [DEV],
            output_device = DEV
        )
    if not LO_RANK:
        summary(net, input_size=(1, 3, 32, 32), device=DEV)
    opt = SGD(net.parameters(), LR, SGD_MOM)
    sched = CosineAnnealingLR(opt, T_max=T_MAX)
    max_te_acc = 0
    for i in range(N_EPOCHS):
        net.train()
        if IS_DISTR:
            l_tr.sampler.set_epoch(i)
        tr_loss, tr_acc, tr_dur = propagate_epoch(net, opt, l_tr, i)
        if IS_DISTR:
            torch.cuda.synchronize()

        max_te_acc = evaluate(
            net, l_te, i, tr_loss, tr_acc, tr_dur, max_te_acc
        )
        sched.step()
    if IS_DISTR:
        torch.cuda.synchronize()
        barrier()
        destroy_process_group()

main()
