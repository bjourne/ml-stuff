# Copyright (C) 2024 Björn Lindqvist
#
# Code demonstrates how to convert an ANN for MNIST into an SNN by
# replacing ReLUs with IF neurons.
from pathlib import Path
from torchinfo import summary
from torch.nn.functional import cross_entropy, linear, relu
from torch.nn import *
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import *

import torch

BS = 512
DATA_PATH = Path('/tmp/data')
NET_PATH = DATA_PATH / 'net.pth'
N_EPOCHS = 3
N_TIME_STEPS = 32
LOG_INT = 100
V_THR = 64
N_CLS = 10

ANN = Sequential(
    Flatten(),
    Linear(784, 64),
    ReLU(),
    Dropout(0.1),
    Linear(64, 64),
    ReLU(),
    Dropout(0.1),
    Linear(64, N_CLS)
)

def if_forward(v, x):
    # Model performs better if we allow negative x.
    v += x
    v = torch.clamp(v, min = 0)
    x = (v >= V_THR).float() * V_THR
    v = v * (v < V_THR)
    return v, x

def snn(x, w1, b1, w2, b2, w3, b3):
    n1 = torch.zeros((BS, 64))
    n2 = torch.zeros((BS, 64))
    y2 = torch.zeros((BS, N_CLS))
    for _ in range(N_TIME_STEPS):
        xp = (torch.rand(x.size()) < x).float()
        xp = torch.reshape(xp, (xp.size(0), -1))
        xp = linear(xp, w1, b1)
        n1, xp = if_forward(n1, xp)
        xp = linear(xp, w2, b2)
        n2, xp = if_forward(n2, xp)
        y2 += linear(xp, w3, b3)
    return y2 / N_TIME_STEPS

def compute_acc(y, yh):
    return (yh.argmax(1) == y).sum().item() / y.size(0)

def propagate_epoch(net, opt, loader, epoch):
    phase = "train" if net.training else "test"
    args = phase, epoch, N_EPOCHS
    print("== %s %3d/%3d ==" % args)
    tot_loss = 0
    tot_acc = 0
    for i, (x, y) in enumerate(loader, n):
        if net.training:
            opt.zero_grad()
        yh = net(x)
        loss = cross_entropy(yh, y)
        if net.training:
            loss.backward()
            opt.step()
        loss = loss.item()
        acc = compute_acc(y, yh)
        if i % LOG_INT == 0:
            print("%4d/%4d, loss/acc: %.4f/%.2f" % (i, n, loss, acc))
        tot_loss += loss
        tot_acc += acc
    return tot_loss / n, tot_acc / n

def get_loaders():
    trans = Compose([ToTensor()])
    ds = [MNIST(DATA_PATH, mode, trans, None, True) for mode in [True, False]]
    return [DataLoader(d, BS, True, drop_last = True) for d in ds]

def train():
    ls = get_loaders()
    summary(ANN, input_size = (1, 1, 28, 28), device = 'cpu')
    opt = Adam(ANN.parameters(), lr = 0.01, betas = (0.9, 0.999))

    max_te_acc = 0
    for i in range(N_EPOCHS):
        net.train()
        tr_loss, tr_acc = propagate_epoch(ANN, opt, ls[0], i)
        net.eval()
        with torch.no_grad():
            te_loss, te_acc = propagate_epoch(ANN, opt, ls[1], i)
        max_te_acc = max(max_te_acc, te_acc)
        fmt = "losses %5.3f/%5.3f acc %5.3f/%5.3f, best acc %5.3f"
        print(fmt % (tr_loss, te_loss, tr_acc, te_acc, max_te_acc))
    torch.save(net.state_dict(), NET_PATH)

def test():
    ls = get_loaders()
    weights = torch.load(NET_PATH, weights_only = True)
    w1 = weights['1.weight']
    b1 = weights['1.bias']
    w2 = weights['4.weight']
    b2 = weights['4.bias']
    w3 = weights['7.weight']
    b3 = weights['7.bias']
    ANN.load_state_dict(weights)
    ANN.eval()

    loader = get_loaders()[1]
    acc1_tot = 0
    acc2_tot = 0
    for x, y in loader:
        acc1 = compute_acc(y, ANN(x))
        acc2 = compute_acc(y, snn(x, w1, b1, w2, b2, w3, b3))
        print('ANN: %6.3f, SNN: %6.3f' % (acc1, acc2))

        acc1_tot += acc1
        acc2_tot += acc2
    n = len(loader)
    acc1_tot /= n
    acc2_tot /= n
    print('Total: %6.3f %6.3f' % (acc1_tot, acc2_tot))

test()
#train()
