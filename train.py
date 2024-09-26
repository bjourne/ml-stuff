# Copyright (C) 2024 Björn A. Lindqvist
#
# VGG16 and ResNets trained on CIFAR10/100.
from clize import run
from mlstuff import (
    load_cifar, loader_sample_figure,
    propagate_epoch, seed_all
)
from mlstuff.networks import QCFS, load_net
from pathlib import Path
from torch import no_grad
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import torch

N_CLS = 100
BS = 256
DATA_PATH = Path("/tmp/data")
LOG_PATH = Path("/tmp/logs")
LR = 0.1
N_EPOCHS = 500
T_MAX = N_EPOCHS
SGD_MOM = 0.9
PRINT_INTERVAL = 10
SEED = 1001

def write_thetas(writer, net, epoch):
    kvs = {}
    for name, m in net.named_modules():
        if isinstance(m, QCFS):
            kvs[name] = m.theta
    writer.add_scalars('thetas', kvs, epoch)

def train(net_name):
    '''Trains a network

    :param net_name: Name of network to train
    '''
    #dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = 'cpu'
    seed_all(SEED)
    net = load_net(net_name, N_CLS).to(dev)
    summary(net)

    dir = 'runs_%s' % net_name
    writer = SummaryWriter(LOG_PATH / dir)

    l_tr, l_te, names = load_cifar(DATA_PATH, BS, N_CLS, dev)
    opt = SGD(net.parameters(), LR, SGD_MOM)
    sched = CosineAnnealingLR(opt, T_max=T_MAX)
    max_te_acc = 0

    for i in range(N_EPOCHS):
        net.train()
        tr_loss, tr_acc = propagate_epoch(
            net, opt, l_tr, i, N_EPOCHS, PRINT_INTERVAL
        )
        net.eval()
        with no_grad():
            te_loss, te_acc = propagate_epoch(
                net, opt, l_te, i, N_EPOCHS, PRINT_INTERVAL
            )
        max_te_acc = max(max_te_acc, te_acc)
        fmt = "losses %5.3f/%5.3f acc %5.3f/%5.3f, (best %5.3f)"
        print(fmt % (tr_loss, te_loss, tr_acc, te_acc, max_te_acc))

        # Per-net stats
        if net_name == 'vgg16qcfs':
            write_thetas(writer, net, i)
        writer.add_scalars('acc', {'train' : tr_acc, 'test' : te_acc}, i)
        writer.add_scalars('loss', {'train' : tr_loss, 'test' : te_loss}, i)
        writer.add_scalar('lr', sched.get_last_lr()[0], i)
        for label, loader in [('training', l_tr), ('testing', l_te)]:
            fig = loader_sample_figure(loader, names, net)
            writer.add_figure(label, fig, i)
        writer.flush()
        sched.step()

if __name__ == '__main__':
    run(train)
