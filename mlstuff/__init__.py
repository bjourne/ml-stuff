# Copyright (C) 2024 Björn A. Lindqvist
from itertools import islice
from matplotlib import pyplot
from pickle import load
from torch.nn.functional import cross_entropy, mse_loss, one_hot
from torch.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    MaxPool2d,
    Module,
    Parameter,
    ReLU,
    Sequential,
)
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

########################################################################
# Data processing
########################################################################

# Tie the device to the DataLoader
class DevDataLoader(DataLoader):
    def __init__(self, dev, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev = dev

    def __iter__(self):
        for x, y in super().__iter__():
            yield x.to(self.dev), y.to(self.dev)

def load_cifar(data_dir, batch_size, n_cls, dev):
    norm = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trans_tr = Compose([
        RandomCrop(32, padding = 4),
        Cutout(),
        RandomHorizontalFlip(),
        ToTensor(),
        norm
    ])
    trans_te = Compose([ToTensor(), norm])
    cls = CIFAR10 if n_cls == 10 else CIFAR100
    d_tr = cls(data_dir, True, trans_tr, download = True)
    d_te = cls(data_dir, False, trans_te, download = True)
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

# Build feature extraction layers based on spec
def make_layers():
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

########################################################################
# Models
########################################################################
class VGG16(Module):
    def __init__(self, n_cls):
        super(VGG16, self).__init__()
        layers = list(make_layers())
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
