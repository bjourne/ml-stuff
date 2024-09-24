# Copyright (C) 2024 Björn A. Lindqvist
#
# VGG16 from scratch and trained on CIFAR10.
from itertools import islice
from matplotlib import pyplot
from mlstuff import VGG16, load_cifar
from pathlib import Path
from torch import no_grad
from torch.nn.functional import cross_entropy, mse_loss, one_hot
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import numpy as np

N_CLS = 100
BS = 256
DATA_DIR = Path("/tmp/data")
LR = 0.1
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
    tot_loss /= n
    tot_acc /= n
    return tot_loss, tot_acc

def loader_sample_figure(loader, names, net):
    x, y = next(iter(loader))
    yh = net(x)
    correct = (yh.argmax(1) == y)
    x = ((x - x.min()) / (x.max() - x.min()))
    fig = pyplot.figure(figsize=(12, 12))
    s = 4
    for i in range(s*s):
        ax = fig.add_subplot(s, s, i + 1, xticks = [], yticks = [])
        pyplot.imshow(np.transpose(x[i], (1, 2, 0)))
        color = 'green' if correct[i] else 'red'
        ax.set_title(
            names[y[i]],
            fontdict=dict(color=color)
        )
    pyplot.close(fig)
    return fig

def main():
    writer = SummaryWriter(DATA_DIR / 'runs')

    net = VGG16(N_CLS)
    l_tr, l_te, names = load_cifar(DATA_DIR, BS, N_CLS)
    opt = SGD(net.parameters(), LR, SGD_MOM)
    sched = CosineAnnealingLR(opt, T_max=T_MAX)
    max_te_acc = 0

    for i in range(N_EPOCHS):
        net.train()
        tr_loss, tr_acc = propagate_epoch(net, opt, l_tr, i)
        net.eval()
        with no_grad():
            te_loss, te_acc = propagate_epoch(net, opt, l_te, i)
        max_te_acc = max(max_te_acc, te_acc)
        fmt = "losses %5.3f/%5.3f acc %5.3f/%5.3f, max acc %5.3f"
        print(fmt % (tr_loss, te_loss, tr_acc, te_acc, max_te_acc))

        writer.add_scalars('acc', {'train' : tr_acc, 'test' : te_acc}, i)
        writer.add_scalars('loss', {'train' : tr_loss, 'test' : te_loss}, i)
        writer.add_scalar('lr', sched.get_last_lr()[0], i)

        for label, loader in [('training', l_tr), ('testing', l_te)]:
            fig = loader_sample_figure(loader, names, net)
            writer.add_figure(label, fig, i)
        writer.flush()
        sched.step()

main()
