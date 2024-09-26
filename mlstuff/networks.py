# Copyright (C) 2024 Björn A. Lindqvist
from torch.nn import *
from torch.nn.init import constant_, kaiming_normal_, normal_

# No expansion factor yet
class ResNetBasicBlock(Module):
    def __init__(self, n_in, n_out, stride):
        super(ResNetBasicBlock, self).__init__()
        shortcut = []
        if stride != 1 or n_in != n_out:
            shortcut = [
                Conv2d(n_in, n_out, 1, stride, 0, bias = False),
                BatchNorm2d(n_out)
            ]
        self.shortcut = Sequential(*shortcut)
        self.conv1 = Conv2d(n_in, n_out, 3, stride, 1, bias = False)
        self.bn1 = BatchNorm2d(n_out)
        self.relu = ReLU(inplace = True)
        self.conv2 = Conv2d(n_out, n_out, 3, 1, 1, bias = False)
        self.bn2 = BatchNorm2d(n_out)

    def forward(self, orig):
        x = self.conv1(orig)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x + self.shortcut(orig))

def make_resnet_layer(n_res_blocks, n_in, n_out, stride):
    yield ResNetBasicBlock(n_in, n_out, stride)
    for i in range(n_res_blocks - 1):
        yield ResNetBasicBlock(n_out, n_out, 1)

class ResNet(Module):
    def __init__(self, layers, n_cls):
        super(ResNet, self).__init__()

        # Prelude
        self.conv1 = Conv2d(3, 64, 7, 2, 3, bias = False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace = True)
        self.mp = MaxPool2d(3, 2, 1)

        # Layers
        self.layer1 = Sequential(*make_resnet_layer(layers[0], 64, 64, 1))
        self.layer2 = Sequential(*make_resnet_layer(layers[1], 64, 128, 2))
        self.layer3 = Sequential(*make_resnet_layer(layers[2], 128, 256, 2))
        self.layer4 = Sequential(*make_resnet_layer(layers[3], 256, 512, 2))

        # Classifier
        self.ap = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512, n_cls)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mp(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Small ResNet for CIFAR10/100
class ResNet4CIFAR(Module):
    def __init__(self, layers, n_cls):
        super(ResNet4CIFAR, self).__init__()
        self.conv1 = Conv2d(3, 16, 3, 1, 1, bias = False)
        self.bn1 = BatchNorm2d(16)
        self.relu = ReLU(inplace = True)
        self.layer1 = Sequential(*make_resnet_layer(layers[0], 16, 16, 1))
        self.layer2 = Sequential(*make_resnet_layer(layers[1], 16, 32, 2))
        self.layer3 = Sequential(*make_resnet_layer(layers[2], 32, 64, 2))
        self.ap = AvgPool2d((8, 8))
        self.fc = Linear(64, n_cls)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.ap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Build feature extraction layers based on spec
def make_vgg_layers():
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
        layers = list(make_vgg_layers())
        self.features = Sequential(*layers)
        self.classifier = Sequential(
            Flatten(),
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
        return self.classifier(x)

def load_net(net_name, n_cls):
    if net_name == 'vgg16':
        return VGG16(n_cls)
    elif net_name == 'resnet18':
        return ResNet([2, 2, 2, 2], n_cls)
    elif net_name == 'resnet20':
        # Much smaller ResNet variant that may work well on CIFAR10/100
        assert n_cls <= 100
        return ResNet4CIFAR([3, 3, 3], n_cls)
    assert False
