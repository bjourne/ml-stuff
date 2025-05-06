from mlstuff.networks import QCFS, QCFSNetwork, ReLU, Sequential, load_net
from torch.nn import ReLU, Sequential
from torchvision.transforms import Resize

import torch

def test_load_qcfs_network():
    net = load_net("resnet18qcfs", 100, 8, "lbl")
    assert isinstance(net, QCFSNetwork)
    net = load_net("resnet18", 100, 8, "lbl")
    assert not isinstance(net, QCFSNetwork)

def test_qcfs():
    net = Sequential(ReLU())
    net = QCFSNetwork(net, 3.0, 8, 8, "lbl")

    x = torch.rand(8, 20)
    y = net(x)
    assert y.shape == (8, 20)

def test_no_resize():
    net = load_net("alexnet", 100, 8, "lbl")
    assert not isinstance(net[0], Resize)
