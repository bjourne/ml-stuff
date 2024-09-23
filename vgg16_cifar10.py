# Copyright (C) 2024 Björn A. Lindqvist
#
# VGG16 from scratch and trained on CIFAR10.
from itertools import islice
from mlstuff import VGG16, load_cifar10
from pathlib import Path
from torch.nn.functional import cross_entropy, mse_loss, one_hot
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchinfo import summary

import torch

N_CLS = 10
BS = 256
DATA_DIR = Path("/tmp/data")
LR = 0.01
N_EPOCHS = 500
T_MAX = 50
SGD_MOM = 0.9

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
    l_tr, l_te = load_cifar10(DATA_DIR, BS)

    net = VGG16(N_CLS)
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
