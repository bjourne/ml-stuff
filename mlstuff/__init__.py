# Copyright (C) 2024 Björn A. Lindqvist
from itertools import islice
from matplotlib import pyplot
from mlstuff.augment import CIFARPolicy, Cutout2
from os import environ
from pickle import load
from random import seed as rseed
from torch.nn.functional import cross_entropy, mse_loss, one_hot
from torch.nn import *
from torch.nn.init import constant_, kaiming_normal_, normal_
from torch.utils.data import DataLoader
from torchtoolbox.transform import Cutout
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

import numpy as np
import torch

########################################################################
# Utils
########################################################################
def seed_all(seed):
    rseed(seed)
    environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

########################################################################
# Data processing
########################################################################
def transforms_aa():
    # These values should be good for CIFAR100
    norm = Normalize(
        [n/255. for n in [129.3, 124.1, 112.4]],
        [n/255. for n in [68.2,  65.4,  70.4]]
    )
    tr  = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        CIFARPolicy(),
        ToTensor(),
        norm,
        Cutout2(n_holes=1, length=16)
    ])
    te = Compose([
        ToTensor(),
        norm
    ])
    return tr, te

def transforms_std():
    norm = Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
    tr = Compose([
        RandomCrop(32, padding = 4),
        Cutout(),
        RandomHorizontalFlip(),
        ToTensor(),
        norm
    ])
    te = Compose([ToTensor(), norm])
    return tr, te

# Tie the device to the DataLoader
class DevDataLoader(DataLoader):
    def __init__(self, dev, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev = dev

    def __iter__(self):
        for x, y in super().__iter__():
            yield x.to(self.dev), y.to(self.dev)

def load_cifar(data_dir, batch_size, n_cls, dev):
    t_tr, t_te = transforms_aa()
    cls = CIFAR10 if n_cls == 10 else CIFAR100
    d_tr = cls(data_dir, True, t_tr, download = True)
    d_te = cls(data_dir, False, t_te, download = True)
    l_tr = DevDataLoader(dev, d_tr, batch_size, True, drop_last = True)
    l_te = DevDataLoader(dev, d_te, batch_size, True, drop_last = True)
    if n_cls == 10:
        names = []
    else:
        meta = data_dir / 'cifar-100-python' / 'meta'
        with open(meta, 'rb') as f:
            d = load(f)
            names = d['fine_label_names']

    return l_tr, l_te, names

########################################################################
# Models
########################################################################

# No expansion factor yet
class ResNetBasicBlock(Module):
    def __init__(self, n_in, n_out, stride):
        super(ResNetBasicBlock, self).__init__()
        shortcut = []
        if stride != 1 or n_in != n_out:
            shortcut = [
                Conv2d(n_in, n_out, 1, stride, 0, bias = False),
                BatchNorm2d(n_out)
            ]
        self.shortcut = Sequential(*shortcut)
        self.conv1 = Conv2d(n_in, n_out, 3, stride, 1, bias = False)
        self.bn1 = BatchNorm2d(n_out)
        self.relu = ReLU(inplace = True)
        self.conv2 = Conv2d(n_out, n_out, 3, 1, 1, bias = False)
        self.bn2 = BatchNorm2d(n_out)

    def forward(self, orig):
        x = self.conv1(orig)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x + self.shortcut(orig))

def make_resnet_layer(n_res_blocks, n_in, n_out, stride):
    yield ResNetBasicBlock(n_in, n_out, stride)
    for i in range(n_res_blocks - 1):
        yield ResNetBasicBlock(n_out, n_out, 1)

class ResNet(Module):
    def __init__(self, layers, n_cls):
        super(ResNet, self).__init__()

        # Prelude
        self.conv1 = Conv2d(3, 64, 7, 2, 3, bias = False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace = True)
        self.mp = MaxPool2d(3, 2, 1)

        # Layers
        self.layer1 = Sequential(*make_resnet_layer(layers[0], 64, 64, 1))
        self.layer2 = Sequential(*make_resnet_layer(layers[1], 64, 128, 2))
        self.layer3 = Sequential(*make_resnet_layer(layers[2], 128, 256, 2))
        self.layer4 = Sequential(*make_resnet_layer(layers[3], 256, 512, 2))

        # Classifier
        self.ap = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512, n_cls)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mp(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Small ResNet for CIFAR10/100
class ResNet4CIFAR(Module):
    def __init__(self, layers, n_cls):
        super(ResNet4CIFAR, self).__init__()
        self.conv1 = Conv2d(3, 16, 3, 1, 1, bias = False)
        self.bn1 = BatchNorm2d(16)
        self.relu = ReLU(inplace = True)
        self.layer1 = Sequential(*make_resnet_layer(layers[0], 16, 16, 1))
        self.layer2 = Sequential(*make_resnet_layer(layers[1], 16, 32, 2))
        self.layer3 = Sequential(*make_resnet_layer(layers[2], 32, 64, 2))
        self.ap = AvgPool2d((8, 8))
        self.fc = Linear(64, n_cls)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.ap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Build feature extraction layers based on spec
def make_vgg_layers():
    VGG16_LAYERS = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M",
    ]
    n_chans_in = 3
    for v in VGG16_LAYERS:
        if type(v) == int:
            # Batch norm so no bias.
            yield Conv2d(n_chans_in, v, 3, padding=1, bias=False)
            yield BatchNorm2d(v)
            yield ReLU(True)
            n_chans_in = v
        elif v == "M":
            yield MaxPool2d(2)
        else:
            assert False

class VGG16(Module):
    def __init__(self, n_cls):
        super(VGG16, self).__init__()
        layers = list(make_vgg_layers())
        self.features = Sequential(*layers)
        self.classifier = Sequential(
            Linear(512, 4096),
            ReLU(),
            Dropout(0.5),
            Linear(4096, 4096),
            ReLU(),
            Dropout(0.5),
            Linear(4096, n_cls),
        )
        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
            elif isinstance(m, Linear):
                normal_(m.weight, 0, 0.01)
                constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_net(net_name, n_cls):
    if net_name == 'vgg16':
        return VGG16(n_cls)
    elif net_name == 'resnet18':
        return ResNet([2, 2, 2, 2], n_cls)
    elif net_name == 'resnet20':
        # Much smaller ResNet variant that may work well on CIFAR10/100
        assert n_cls <= 100
        return ResNet4CIFAR([3, 3, 3], n_cls)
    assert False

########################################################################
# Training
########################################################################
def propagate_epoch(net, opt, loader, epoch, n_epochs, print_interval):
    phase = "train" if net.training else "test"
    args = phase, epoch, n_epochs
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
        acc = (yh.argmax(1) == y).sum().item() / y.size(0)
        if i % print_interval == 0:
            print("%4d/%4d, loss/acc: %.4f/%.2f" % (i, n, loss, acc))
        tot_loss += loss
        tot_acc += acc
    tot_loss /= n
    tot_acc /= n
    return tot_loss, tot_acc

########################################################################
# Training
########################################################################
def loader_sample_figure(loader, names, net):
    x, y = next(iter(loader))
    yh = net(x)
    correct = (yh.argmax(1) == y)
    x = ((x - x.min()) / (x.max() - x.min()))
    fig = pyplot.figure(figsize=(12, 12))
    s = 4
    for i in range(s*s):
        ax = fig.add_subplot(s, s, i + 1, xticks = [], yticks = [])
        pyplot.imshow(np.transpose(x[i].cpu(), (1, 2, 0)))
        color = 'green' if correct[i] else 'red'
        ax.set_title(
            names[y[i]],
            fontdict=dict(color=color)
        )
    pyplot.close(fig)
    return fig

def main():
    from torchinfo import summary
    net = load_net('resnet20', 100)
    summary(net, input_size=(1, 3, 32, 32), device="cpu")

if __name__ == '__main__':
    main()
