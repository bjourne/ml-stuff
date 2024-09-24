# Copyright (C) 2024 Björn A. Lindqvist
from pickle import load
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
from torch.utils.data import DataLoader
from torchtoolbox.transform import Cutout
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

def load_cifar(data_dir, batch_size, n_cls):
    norm = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trans_tr = Compose([
        RandomCrop(32, padding = 4),
        Cutout(),
        RandomHorizontalFlip(),
        ToTensor(),
        norm
    ])
    trans_te = Compose([ToTensor(), norm])
    if n_cls == 10:
        cls = CIFAR10
        names = []
    else:
        cls = CIFAR100
        meta = data_dir / 'cifar-100-python' / 'meta'
        with open(meta, 'rb') as f:
            d = load(f)
            names = d['fine_label_names']


    d_tr = cls(data_dir, True, trans_tr, download = True)
    d_te = cls(data_dir, False, trans_te, download = True)
    l_tr = DataLoader(d_tr, batch_size, True, drop_last = True)
    l_te = DataLoader(d_te, batch_size, True, drop_last = True)
    return l_tr, l_te, names

# Build feature extraction layers based on spec
def make_layers():
    VGG16_LAYERS = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M",
    ]
    n_chans_in = 3
    for v in VGG16_LAYERS:
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

class VGG16(Module):
    def __init__(self, n_cls):
        super(VGG16, self).__init__()
        layers = list(make_layers())
        self.features = Sequential(*layers)
        self.classifier = Sequential(
            Linear(512, 4096),
            ReLU(),
            Dropout(0.5),
            Linear(4096, 4096),
            ReLU(),
            Dropout(0.5),
            Linear(4096, n_cls),
        )
        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    constant_(m.bias, 0)
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
