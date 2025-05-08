from mlstuff import param_count
from mlstuff.networks import (
    EfficientNet, QCFSNetwork, ReLU, Sequential,
    load_base_net, load_net
)
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

def test_count_params():
    net = load_base_net("resnet18", 100)
    assert param_count(net) == 11220132
    net = load_base_net("resnet18pa", 100)
    assert param_count(net) == 11220132

    assert param_count(EfficientNet("b0", 10)) == 4020358
    assert param_count(EfficientNet("b1", 10)) == 6525994
    assert param_count(EfficientNet("b2", 10)) == 7715084
    assert param_count(EfficientNet("b3", 10)) == 10711602
    assert param_count(EfficientNet("b4", 10)) == 17566546
    assert param_count(EfficientNet("b5", 10)) == 28361274
    assert param_count(EfficientNet("b6", 10)) == 40758754
