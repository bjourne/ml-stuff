# Copyright (C) 2024 Björn A. Lindqvist
from itertools import islice
from matplotlib import pyplot
from mlstuff.augment import CIFARPolicy, Cutout2
from os import environ
from pickle import load
from random import seed as rseed
from re import sub
from torch.nn.functional import cross_entropy, mse_loss
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

BU2023_ENDINGS = {
    'vgg16qcfs' : [
        ('.thresh', '.theta')
    ],
}

BU2023_STARTS = {
    'vgg16qcfs' : [
        ('layer1.2', 'features.2'),
        ('layer1.6', 'features.5'),
        ('layer2.2', 'features.9'),
        ('layer2.6', 'features.12'),
        ('layer3.2', 'features.16'),
        ('layer3.6', 'features.19'),
        ('layer3.10', 'features.22'),
        ('layer4.2', 'features.26'),
        ('layer4.6', 'features.29'),
        ('layer4.10', 'features.32'),
        ('layer5.2', 'features.36'),
        ('layer5.6', 'features.39'),
        ('layer5.10', 'features.42'),
        ('classifier.2', 'classifier.2'),
        ('classifier.5', 'classifier.4'),
        ('layer1.0', 'features.0'),
        ('layer1.1', 'features.1'),
        ('layer1.4', 'features.3'),
        ('layer1.5', 'features.4'),
        ('layer2.0', 'features.7'),
        ('layer2.1', 'features.8'),
        ('layer2.4', 'features.10'),
        ('layer2.5', 'features.11'),
        ('layer3.0', 'features.14'),
        ('layer3.1', 'features.15'),
        ('layer3.4', 'features.17'),
        ('layer3.5', 'features.18'),
        ('layer3.8', 'features.20'),
        ('layer3.9', 'features.21'),
        ('layer4.0', 'features.24'),
        ('layer4.1', 'features.25'),
        ('layer4.4', 'features.27'),
        ('layer4.5', 'features.28'),
        ('layer4.8', 'features.30'),
        ('layer4.9', 'features.31'),
        ('layer5.0', 'features.34'),
        ('layer5.1', 'features.35'),
        ('layer5.4', 'features.37'),
        ('layer5.5', 'features.38'),
        ('layer5.6', 'features.39'),
        ('layer5.8', 'features.40'),
        ('layer5.9', 'features.41'),
        ('layer5.10', 'features.42'),
        ('classifier.4', 'classifier.3'),
        ('classifier.7', 'classifier.5')
    ]
}

BU2023_REGEXP_REPLS = {
    'resnet34qcfs' : [
        (r'^conv2_x\.(.*)$', r'layer1.\g<1>'),
        (r'^conv3_x\.(.*)$', r'layer2.\g<1>'),
        (r'^conv4_x\.(.*)$', r'layer3.\g<1>'),
        (r'^conv5_x\.(.*)$', r'layer4.\g<1>'),

        # Residual
        (r'^([^\.]+)\.(\d)\.residual_function\.(\d)\.(\w+)$', r'\g<1>.\g<2>.residual.\g<3>.\g<4>'),

        # Act
        (r'^([^\.]+\.\d).act.([^\.]+)$', r'\g<1>.relu.\g<2>'),

        # Free
        (r'^conv1\.0\.([^\.]+)$', r'conv1.\g<1>'),
        (r'^conv1\.1\.([^\.]+)$', r'bn1.\g<1>'),
        (r'^conv1\.2\.([^\.]+)$', r'relu.\g<1>'),

        # Theta
        (r'^([^\.]+|[^\.]+\.\d\.[^\.]+|[^\.]+\.\d\.[^\.]+\.\d)\.thresh$',
         r'\g<1>.theta'),
    ]
}

BU2023_REGEXP_REPLS['resnet18qcfs'] = BU2023_REGEXP_REPLS['resnet34qcfs']
BU2023_REGEXP_REPLS['resnet20qcfs'] = BU2023_REGEXP_REPLS['resnet34qcfs']

def rename_bu2023(net_name, d):
    repls = BU2023_REGEXP_REPLS[net_name]
    d2 = {}
    for k, v in d.items():
        for pattern, repl in repls:
            k = sub(pattern, repl, k)
        d2[k] = v
    return d2

    # endings = BU2023_ENDINGS[net_name]
    # d2 = {}
    # for k, v in d.items():
    #     k2 = k
    #     for src, dst in endings:
    #         if k.endswith(src):
    #             k2 = k[:-len(src)] + dst
    #             break
    #     d2[k2] = v

    # starts = BU2023_STARTS[net_name]
    # d3 = {}
    # for k, v in d2.items():
    #     k2 = k
    #     for src, dst in starts:
    #         if k.startswith(src):
    #             k2 = dst + k[len(src):]
    #             break
    #     d3[k2] = v
    # return d3

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
    l_te = DevDataLoader(dev, d_te, batch_size, False, drop_last = True)
    if n_cls == 10:
        names = []
    else:
        meta = data_dir / 'cifar-100-python' / 'meta'
        with open(meta, 'rb') as f:
            d = load(f)
            names = d['fine_label_names']

    return l_tr, l_te, names

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
        n_corr = (yh.argmax(1) == y).sum().item()
        acc = n_corr / y.size(0)
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
    d = {
        'conv5_x.2.act.thresh' : None,
        'conv5_x.2.residual_function.4.running_mean' : None,
        'relu.thresh' : None,
        'conv1.0.weight' : None
    }
    d = rename_bu2023('resnet34qcfs', d)
    print(d)
