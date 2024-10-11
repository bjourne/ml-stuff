# Copyright (C) 2024 Björn A. Lindqvist
#
# VGG16 and ResNets trained on CIFAR10/100.
from clize import run
from mlstuff import (
    load_cifar, loader_sample_figure,
    propagate_epoch, rename_bu2023, seed_all
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
DATA_PATH = Path("/tmp/data")
LOG_PATH = Path("/tmp/logs")
LR = 0.1
N_EPOCHS = 600
T_MAX = N_EPOCHS
SGD_MOM = 0.9
PRINT_INTERVAL = 30
SEED = 1001

def write_thetas(writer, net, epoch):
    kvs = {}
    for name, m in net.named_modules():
        if isinstance(m, QCFS):
            kvs[name] = m.theta
    writer.add_scalars('thetas', kvs, epoch)

def train(net_name, batch_size: int, weight_decay: float):
    '''Trains a network

    :param net_name: Name of network to train
    :param batch_size: Batch size
    '''
    dev = 'cpu'
    net = load_net(net_name, N_CLS).to(dev)
    summary(net)

    dir_name = 'runs_%s_%03d_%.4f' % (net_name, batch_size, weight_decay)
    out_dir = LOG_PATH / dir_name
    writer = SummaryWriter(out_dir)

    l_tr, l_te, names = load_cifar(DATA_PATH, batch_size, N_CLS, dev)
    opt = SGD(net.parameters(), LR, SGD_MOM, weight_decay = weight_decay)
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
        if te_acc > max_te_acc:
            torch.save(net.state_dict(), out_dir / 'net.pth')

        max_te_acc = max(max_te_acc, te_acc)
        fmt = "losses %5.3f/%5.3f acc %5.3f/%5.3f, (best %5.3f)"
        print(fmt % (tr_loss, te_loss, tr_acc, te_acc, max_te_acc))

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

def test(net_name, batch_size: int, weights_file, *, time: int = None):
    '''Tests a network

    :param net_name: Name of network to train
    :param batch_size: Batch size
    :param weights_file: Path to weights file
    :param time: Number of simulation time steps (for vgg16qcfs)
    '''
    dev = 'cpu'
    net = load_net(net_name, N_CLS).to(dev)
    d = torch.load(weights_file, weights_only = True, map_location = 'cpu')
    if net_name.endswith('qcfs'):
        d = rename_bu2023(net_name, d)
        net.set_snn_mode(time)
        net.net.load_state_dict(d)
    else:
        net.load_state_dict(d)

    _, l_te, names = load_cifar(DATA_PATH, batch_size, N_CLS, dev)
    net.eval()
    with no_grad():
        te_loss, te_acc = propagate_epoch(
            net, None, l_te, 0, 1, PRINT_INTERVAL
        )
    print("loss %5.3f, acc %5.3f" % (te_loss, te_acc))

if __name__ == '__main__':
    seed_all(SEED)
    run(train, test)
