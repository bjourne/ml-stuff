from mlstuff.networks import QCFS, QCFSNetwork, load_net
from torch.nn import ReLU, Sequential

def test_load_qcfs_network():
    net = load_net("resnet18qcfs", 100, 8, "lbl")
    assert isinstance(net, QCFSNetwork)
    net = load_net("resnet18", 100, 8, "lbl")
    assert not isinstance(net, QCFSNetwork)
