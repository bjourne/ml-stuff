# Copyright (C) 2024 Björn A. Lindqvist
#
# VGG16 from scratch and trained on CIFAR10.
from itertools import islice
from pathlib import Path
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

import torch

N_CLS = 10
BS = 128
DATA_DIR = Path("/tmp/data")
LR = 0.01
N_EPOCHS = 500
T_MAX = 50
SGD_MOM = 0.9

# Build feature extraction layers based on spec
def make_layers(spec):
    n_chans_in = 3
    for v in spec:
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


FEATURE_LAYERS = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M",
]

# The accuracy of the mode lis 93.6% with a normal top, but only 92.1%
# with a Linear(512, N_CLS) top.
class VGG16(Module):
    def __init__(self):
        super(VGG16, self).__init__()
        layers = list(make_layers(FEATURE_LAYERS))
        self.features = Sequential(*layers)
        self.classifier = Sequential(
            Linear(512, 4096),
            ReLU(),
            Dropout(0.5),
            Linear(4096, 4096),
            ReLU(),
            Dropout(0.5),
            Linear(4096, N_CLS),
        )
        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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

    net = VGG16()
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
