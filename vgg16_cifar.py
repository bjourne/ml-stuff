# Copyright (C) 2024 Björn A. Lindqvist
#
# VGG16 from scratch and trained on CIFAR10.
from mlstuff import VGG16, load_cifar, loader_sample_figure, propagate_epoch
from pathlib import Path
from torch import no_grad
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import torch

N_CLS = 100
BS = 256
DATA_DIR = Path("/tmp/data")
LR = 0.1
N_EPOCHS = 500
T_MAX = 50
SGD_MOM = 0.9
PRINT_INTERVAL = 10

def main():
    dev = 'cpu'
    #dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(DATA_DIR / 'runs')

    net = VGG16(N_CLS)
    net.to(dev)
    l_tr, l_te, names = load_cifar(DATA_DIR, BS, N_CLS, dev)
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

        writer.add_scalars('acc', {'train' : tr_acc, 'test' : te_acc}, i)
        writer.add_scalars('loss', {'train' : tr_loss, 'test' : te_loss}, i)
        writer.add_scalar('lr', sched.get_last_lr()[0], i)

        for label, loader in [('training', l_tr), ('testing', l_te)]:
            fig = loader_sample_figure(loader, names, net)
            writer.add_figure(label, fig, i)
        writer.flush()
        sched.step()

main()
