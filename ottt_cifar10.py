# Copyright (C) 2024 Björn Lindqvist
#
# Reimplementation of "Online Training Through Time for Spiking Neural Networks"
#
# Code : https://github.com/pkuxmq/OTTT-SNN
# Paper: https://arxiv.org/abs/2210.04195
from itertools import islice
from pathlib import Path

from rich.table import Table
from rich import print as rprint

from torch.autograd import Function
from torch.nn import (
    AvgPool2d,
    Conv2d,
    Linear,
    Module,
    Parameter,
    Sequential,
    init
)
from torch.nn.functional import (
    conv2d,
    cross_entropy,
    mse_loss, one_hot
)
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchtoolbox.transform import Cutout

from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop, RandomHorizontalFlip,
    ToTensor
)

import numpy as np
import random
import torch

_seed_ = 2022
random.seed(_seed_)
torch.manual_seed(_seed_)
np.random.seed(_seed_)

N_EPOCHS = 300
N_CLS = 10
BS = 128
DATA_DIR = Path('/tmp/data')
LOG_DIR = Path('/tmp/logs')

# Simulation settings
N_T_STEPS = 6
TAU = 2.0
ALPHA = 4.0
V_THRESH = 1.0
LEAK_LAMBDA = (1 - 1 / TAU)

# Optimization settings
LR = 0.1
SGD_MOM = 0.9
T_MAX = 300
LOSS_LAMBDA = 0.05

class sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sgax = (x * ALPHA).sigmoid_()
        grad_x = grad_output * (1 - sgax) * sgax * ALPHA
        return grad_x,

class OnlineLIFNode(Module):
    def forward(self, x, init):
        if init:
            self.v = torch.zeros_like(x)
            self.rate = torch.zeros_like(x)

        # Decay and add input
        self.v = self.v.detach() * LEAK_LAMBDA + x

        # Check spiking
        spike = sigmoid.apply(self.v - V_THRESH)

        # Maybe soft reset
        spike_d = spike.detach()
         self.v = self.v - spike_d * V_THRESH

        with torch.no_grad():
            spike_d = spike.clone().detach()
            self.rate = self.rate * LEAK_LAMBDA + spike_d

        if self.training:
            return torch.cat((spike, self.rate))
        return spike

class Replace(Function):
    @staticmethod
    def forward(ctx, z1, z1_r):
        return z1_r

    @staticmethod
    def backward(ctx, grad):
        return grad, grad

class WrappedSNNOp(Module):
    def __init__(self, op):
        super(WrappedSNNOp, self).__init__()
        self.op = op

    def forward(self, x):
        if self.training:
            B = x.shape[0] // 2
            spike = x[:B]
            rate = x[B:]
            with torch.no_grad():
                out = self.op(spike).detach()
            in_for_grad = Replace.apply(spike, rate)
            out_for_grad = self.op(in_for_grad)
            return Replace.apply(out_for_grad, out)
        return self.op(x)

class SequentialModule(Sequential):
    def __init__(self, *args):
        super(SequentialModule, self).__init__(*args)

    def forward(self, x, init):
        for mod in self._modules.values():
            if isinstance(mod, OnlineLIFNode):
                x = mod(x, init)
                # Scaled weight standardization (see Section 4.4 and
                # Appendix C.1).
                x = x * 2.74
            else:
                x = mod(x)
        return x

class ScaledWSConv2d(Conv2d):
    def __init__(self, in_channels, out_channels):
        super(ScaledWSConv2d, self).__init__(
            in_channels, out_channels,
            3, 1, 1, 1,
            1, True
        )
        self.gain = Parameter(torch.ones(self.out_channels, 1, 1, 1))

    def forward(self, x):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + 1e-4) ** 0.5)
        weight = weight * self.gain
        return conv2d(
            x, weight,
            self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )

class OnlineSpikingVGG(Module):
    def __init__(self):
        super(OnlineSpikingVGG, self).__init__()
        self.features = self.make_layers()
        self.classifier = WrappedSNNOp(Linear(512, N_CLS))
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

    def forward(self, x, init):
        x = self.features(x, init)
        x = torch.mean(x, dim = (2, 3))
        return self.classifier(x)

    @staticmethod
    def make_layers():
        layers = []
        n_chan_in = 3
        first_conv = True

        # Only one VGG arch
        cfg = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
        for v in cfg:
            if v == 'M':
                layers += [AvgPool2d(2, 2)]
            elif type(v) == int:
                conv2d = ScaledWSConv2d(n_chan_in, v)
                if not first_conv:
                    conv2d = WrappedSNNOp(conv2d)
                first_conv = False
                layers += [conv2d, OnlineLIFNode()]
                n_chan_in = v
        return SequentialModule(*layers)

def compute_loss(yh, y):
    y_one_hot = one_hot(y, N_CLS).float()
    loss0 = LOSS_LAMBDA * mse_loss(yh, y_one_hot)
    loss1 = (1 - LOSS_LAMBDA) * cross_entropy(yh, y)
    loss = (loss0 + loss1) / N_T_STEPS
    return loss

def propagate_batch(net, x, y):
    loss = 0
    tot_yh = torch.zeros((BS, N_CLS))
    for t in range(N_T_STEPS):
        yh = net(x, t == 0)
        tot_yh += yh.clone().detach()
        ls = compute_loss(yh, y)
        loss += ls.item()
        if net.training:
            ls.backward()
    acc = (tot_yh.argmax(1) == y).float().sum().item() / BS
    return loss, acc

def propagate_all(net, opt, loader, epoch, writer):
    phase = 'train' if net.train else 'test'
    args = phase, epoch, N_EPOCHS
    print('== %s %3d/%3d ==' % args)

    tot_loss = 0
    tot_acc = 0
    n = len(loader)
    for i, (x, y) in enumerate(islice(loader, n)):
        if net.train:
            opt.zero_grad()
        loss, acc = propagate_batch(net, x, y)
        if net.train:
            opt.step()
        print('%4d/%4d, loss/acc: %.4f/%.2f' % (i, n, loss, acc))
        tot_loss += loss
        tot_acc += acc
    tot_loss /= n
    tot_acc /= n
    writer.add_scalar('%s_loss' % phase, tot_loss, epoch)
    writer.add_scalar('%s_acc' % phase, tot_acc, epoch)
    return tot_loss, tot_acc

def main():
    trans_tr = Compose([
        RandomCrop(32, padding=4),
        Cutout(),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trans_te = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    d_tr = CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=trans_tr
    )
    l_tr = DataLoader(d_tr, batch_size=BS, shuffle=True, drop_last=True)
    d_te = CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=trans_te
    )
    l_te = DataLoader(d_te, batch_size=BS, shuffle=True, drop_last=True)

    tab = Table('Parameter', 'Value', title = 'Parameters')
    params = [
        ('Batch size', BS),
        ('N training batches', len(l_tr)),
        ('V threshold', V_THRESH),
        ('Leak lambda', LEAK_LAMBDA)
    ]
    for n, v in params:
        tab.add_row(n, str(v))
    rprint(tab)

    net = OnlineSpikingVGG()
    print('Total Parameters: %.2fM'
          % (sum(p.numel() for p in net.parameters()) / 1000000.0))

    opt = SGD(net.parameters(), LR, SGD_MOM)
    lr_scheduler = CosineAnnealingLR(opt, T_max=T_MAX)
    max_test_acc = 0
    writer = SummaryWriter(LOG_DIR)

    for epoch in range(N_EPOCHS):
        net.train()
        train_loss, train_acc = propagate_all(net, opt, l_tr, epoch, writer)
        lr_scheduler.step()

        net.eval()
        test_loss, test_acc = propagate_all(net, opt, l_te, epoch, writer)

        save_max = test_acc > max_test_acc
        max_test_acc = max(max_test_acc, test_acc)
        checkpoint = {
            'net': net.state_dict(),
            'opt': opt.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, LOG_DIR / 'checkpoint_max.pth')

        torch.save(checkpoint, LOG_DIR / 'checkpoint_latest.pth')

        fmt = 'losses %5.3f/%5.3f acc %5.3f/%5.3f, best acc %5.3f'
        print(fmt % (train_loss, test_loss,
                     train_acc, test_acc,
                     max_test_acc))

if __name__ == '__main__':
    main()
