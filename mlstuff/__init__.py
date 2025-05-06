# Copyright (C) 2024-2025 Björn A. Lindqvist <bjourne@gmail.com>
from itertools import islice
from matplotlib import pyplot
from mlstuff.augment import CIFARPolicy, Cutout2
from os import environ
from pickle import load
from random import seed as rseed
from re import sub
from time import time
from torch.distributed import barrier
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
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

def vgg16_layer_repl(m):
    old_idx = int(m.group(1)), int(m.group(2))
    d = {
        (1, 0) : 0, (1, 1) : 1, (1, 2) : 2, (1, 4) : 3, (1, 5) : 4,
        (1, 6) : 5,
        (2, 0) : 7, (2, 1) : 8, (2, 2) : 9, (2, 4) : 10, (2, 5) : 11,
        (2, 6) : 12,
        (3, 0) : 14, (3, 1) : 15, (3, 2) : 16, (3, 4) : 17, (3, 5) : 18,
        (3, 6) : 19, (3, 8) : 20, (3, 9) : 21, (3, 10) : 22,
        (4, 0) : 24, (4, 1) : 25, (4, 2) : 26, (4, 4) : 27, (4, 5) : 28,
        (4, 6) : 29, (4, 8) : 30, (4, 9) : 31, (4, 10) : 32,
        (5, 0) : 34, (5, 1) : 35, (5, 2) : 36, (5, 4) : 37, (5, 5) : 38,
        (5, 6) : 39, (5, 8) : 40, (5, 9) : 41, (5, 10) : 42
    }
    new_idx = d[old_idx]
    rem = m.group(3)
    return '%d.%s' % (new_idx, rem)

def vgg16_classifier_repl(m):
    d = {1 : 45, 2 : 46, 4 : 47, 5 : 48, 7 : 49}
    old_idx = int(m.group(1))
    rem = m.group(2)
    new_idx = d[old_idx]
    return "%d.%s" % (new_idx, rem)


# Order is important
BU2023_REGEXP_REPLS = {
    'vgg16qcfs' : [
        (r'^layer(\d)\.(\d+)\.([^\.]+)$', vgg16_layer_repl),
        (r"^classifier\.(\d)\.([^\.]+)$", vgg16_classifier_repl),
        (r'^(.*)\.thresh$', r'\g<1>.theta')
    ],
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

########################################################################
# For distributed training
########################################################################
def get_device():
    lo_rank = environ.get("LOCAL_RANK")
    if lo_rank is not None:
        return int(lo_rank)
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def is_primary(dev):
    return dev in {0, "cuda", "cpu"}

def is_distributed(dev):
    return type(dev) == int

def synchronize(dev):
    if is_distributed(dev):
        torch.cuda.synchronize()
        barrier()

def identify_worker():
    dev = get_device()
    if is_distributed(dev):
        n_devs = int(environ.get("LOCAL_WORLD_SIZE"))
        print("Worker %d of %d" % (dev, n_devs))
    else:
        print("Running on %s device" % dev)

########################################################################
# Data processing
########################################################################

# Transforms are dependent on both the dataset and the network
def transforms(ds_name, net_name):
    is_cifar = ds_name in {"cifar10", "cifar100"}
    if is_cifar:
        tr_steps, te_steps = [], []

        net_sizes = {
            "alexnet" : 227,
            #"convnext_tiny" : 224
        }
        size = net_sizes.get(net_name, 32)
        if size != 32:
            print("Resizing to %d" % size)
            resize = Resize(size)
            tr_steps.append(resize)
            te_steps.append(resize)
        norm = Normalize(
            [n/255. for n in [129.3, 124.1, 112.4]],
            [n/255. for n in [68.2,  65.4,  70.4]]
        )
        tr_steps.extend([
            RandomCrop(size, padding = size // 8),
            RandomHorizontalFlip(),
            CIFARPolicy(),
            ToTensor(),
            norm,
            Cutout2(n_holes=1, length = size // 2)
        ])
        te_steps.extend([
            ToTensor(), norm
        ])
    else:
        assert False
    return Compose(tr_steps), Compose(te_steps)

# Tie the device to the DataLoader
class DevDataLoader(DataLoader):
    def __init__(self, dev, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev = dev

    def __iter__(self):
        for x, y in super().__iter__():
            yield x.to(self.dev), y.to(self.dev)

def load_data(ds_name, net_name, data_path, batch_size, dev):
    cls = {"cifar10" : CIFAR10, "cifar100" : CIFAR100}[ds_name]

    t_tr, t_te = transforms(ds_name, net_name)
    if is_primary(dev):
        cls(data_path, True, t_tr, download = True)
        cls(data_path, False, t_te, download = True)
    synchronize(dev)
    d_tr = cls(data_path, True, t_tr, download = False)
    d_te = cls(data_path, False, t_te, download = False)

    sampler = None
    shuffle = True
    num_workers = 8
    if is_distributed(dev):
        sampler = DistributedSampler(dataset=d_tr)
        shuffle = False
        num_workers = 16
    l_tr = DevDataLoader(
        dev, d_tr, batch_size,
        shuffle = shuffle,
        sampler = sampler,
        drop_last = True,
        num_workers = num_workers
    )
    l_te = DevDataLoader(
        dev, d_te, batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = num_workers
    )
    if ds_name == "cifar10":
        names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        ]
    else:
        meta = data_path / 'cifar-100-python' / 'meta'
        with open(meta, 'rb') as f:
            d = load(f)
            names = d['fine_label_names']
    return l_tr, l_te, names

########################################################################
# Training
########################################################################
class EpochStats:
    def __init__(self, n):
        self.loss = 0
        self.acc = 0
        self.n = n
        self.bef = time()

    def update(self, loss, acc):
        self.loss += loss
        self.acc += acc

    def finalize(self):
        self.loss /= self.n
        self.acc /= self.n
        self.dur = time() - self.bef

def forward_batch(net, opt, x, y):
    if net.training:
        opt.zero_grad()
    yh = net(x)
    loss = cross_entropy(yh, y)
    if net.training:
        loss.backward()
        opt.step()
    n_corr = (yh.argmax(1) == y).sum().item()
    acc = n_corr / y.size(0)
    return loss.item(), acc

def propagate_epoch(dev, net, opt, loader, epoch, n_epochs, print_interval):
    if is_primary(dev):
        phase = "train" if net.training else "test"
        args = phase, epoch, n_epochs
        print("== %s %3d/%3d ==" % args)
    n = len(loader)
    stats = EpochStats(n)
    for i, (x, y) in enumerate(islice(loader, n)):
        loss, acc = forward_batch(net, opt, x, y)
        if i % print_interval == 0 and is_primary(dev):
            print("%4d/%4d, loss/acc: %.4f/%.2f" % (i, n, loss, acc))
        stats.update(loss, acc)
    stats.finalize()
    return stats

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
        'layer3.1.weight' : None,
    }
    d = rename_bu2023('vgg16qcfs', d)
    print(d)
