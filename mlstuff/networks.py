# Copyright (C) 2024-2025 Björn A. Lindqvist <bjourne@gmail.com>
from math import ceil, floor, sqrt
from torch.autograd import Function
from torch.nn import *
from torch.nn.init import constant_, kaiming_normal_, normal_, uniform_
from torch.nn.functional import relu

import torch

# From https://github.com/putshua/ANN_SNN_QCFS
class GradFloor(Function):
    @staticmethod
    def forward(ctx, x):
        return x.floor()

    @staticmethod
    def backward(ctx, x):
        return x

class QCFS(Module):
    def __init__(self, theta, l):
        super().__init__()
        self.theta = Parameter(torch.tensor([theta]), requires_grad=True)
        self.l = l
        self.n_time_steps = 0

    def forward_qcfs(self, x):
        x = x / self.theta
        x = torch.clamp(x, 0, 1)
        x = GradFloor.apply(x * self.l + 0.5) / self.l
        return x * self.theta

    def forward_if(self, x):
        theta = self.theta.data
        if self.mem is None:
            self.mem = torch.zeros_like(x) + theta * 0.5
        self.mem += x
        x = (self.mem - theta >= 0).float() * theta
        self.mem -= x
        return x

    def forward(self, x):
        if self.n_time_steps > 0:
            return self.forward_if(x)
        return self.forward_qcfs(x)

def replace_modules(mod, match_fun, new_fun):
    for name, submod in mod.named_children():
        if match_fun(submod):
            setattr(mod, name, new_fun(submod))
        replace_modules(submod, match_fun, new_fun)

def is_relu(mod):
    return isinstance(mod, ReLU)

########################################################################
# EfficientNet
########################################################################
EFF_BASE = [
    # expand, n_chans, repeats, stride, k_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]

PHI_VALUES = {
    "b0" : (0.0, 224, 0.2),
    "b1" : (0.5, 240, 0.2),
    "b2" : (1.0, 260, 0.3),
    "b3" : (2.0, 300, 0.3),
    "b4" : (3.0, 380, 0.4),
    "b5" : (4.0, 456, 0.4),
    "b6" : (5.0, 528, 0.5),
    "b7" : (6.0, 600, 0.5)
}

class CNNBlock(Module):
    def __init__(self, n_in, n_out, k_size, stride, padding, groups = 1):
        super().__init__()
        self.conv = Conv2d(
            n_in, n_out, k_size,
            stride, padding,
            groups = groups,
            bias = False
        )
        self.bn = BatchNorm2d(n_out)
        self.silu = SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x

SE_RATIO = 0.25

# Attention scores for each of the channels
class SqueezeExcitation(Module):
    def __init__(self, n_in):
        super(SqueezeExcitation, self).__init__()
        n_reduced = int(n_in * SE_RATIO)
        self.se = Sequential(
            AdaptiveAvgPool2d(1),
            Conv2d(n_in, n_reduced, 1),
            SiLU(),
            Conv2d(n_reduced, n_in, 1),
            Sigmoid()
        )

    def forward(self, x):
        # How can this be element-wise?
        return x * self.se(x)

class InvertedResidualBlock(Module):
    def __init__(
        self, n_in, n_out, k_size, stride, exp_ratio,
        survival_p = 0.8
    ):
        super(InvertedResidualBlock, self).__init__()
        # Check this
        self.survival_p = survival_p
        n_hidden = n_in * exp_ratio
        self.use_residual = n_in == n_out and stride == 1

        self.expand_conv = None
        if exp_ratio != 1:
            self.expand_conv = CNNBlock(
                n_in, n_hidden,
                k_size=1, stride=1, padding=0
            )
        # Depthwise conv
        self.conv = Sequential(
            CNNBlock(
                n_hidden,
                n_hidden,
                k_size,
                stride,
                # Use same padding
                k_size // 2,
                groups = n_hidden
            ),
            SqueezeExcitation(n_hidden),
            Conv2d(n_hidden, n_out, 1, bias = False),
            BatchNorm2d(n_out)
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        # Weird dropout with scaling
        bin_tensor = torch.rand(x.shape[0], 1, 1, 1, device = x.device)
        bin_tensor = bin_tensor < self.survival_p
        return torch.div(x, self.survival_p) * bin_tensor

    def forward(self, x):
        xp = self.expand_conv(x) if self.expand_conv else x
        xp = self.conv(xp)
        if self.use_residual:
            return self.stochastic_depth(xp) + x
        return xp



class EfficientNet(Module):
    def __init__(self, version, n_classes):
        super().__init__()

        # Width, depth, and dropout
        phi, _, dropout = PHI_VALUES[version]
        depth_factor = 1.2**phi
        width_factor = 1.1**phi
        last_channels = ceil(1280 * width_factor)
        self.pool = AdaptiveAvgPool2d(1)
        self.features = self.create_features(
            width_factor,
            depth_factor,
            last_channels
        )
        self.classifier = Sequential(
            Dropout(dropout),
            Linear(last_channels, n_classes)
        )

        # Comes from torchvision
        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, (BatchNorm2d, GroupNorm)):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
            elif isinstance(m, Linear):
                init_range = 1.0 / sqrt(m.out_features)
                uniform_(m.weight, -init_range, init_range)
                constant_(m.bias, 0)



    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride = 2, padding = 1)]
        n_in = channels
        for exp_ratio, channels, n_reps, stride, k_size in EFF_BASE:
            n_out = 4 * ceil(int(channels * width_factor) / 4)
            layers_n_reps = ceil(n_reps * depth_factor)
            for layer in range(layers_n_reps):
                features.append(
                    InvertedResidualBlock(
                        n_in, n_out,
                        stride = stride if layer == 0 else 1,
                        exp_ratio = exp_ratio,
                        k_size = k_size
                    )
                )
                n_in = n_out
        features.append(
            CNNBlock(n_in, last_channels, k_size = 1, stride = 1, padding = 0)
        )
        return Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))








########################################################################
# ResNets
########################################################################

# No expansion factor yet
class ResNetBasicBlock(Module):
    def __init__(self, n_in, n_out, stride):
        super().__init__()
        self.residual = Sequential(
            Conv2d(n_in, n_out, 3, stride, 1, bias = False),
            BatchNorm2d(n_out),
            ReLU(inplace = True),
            Conv2d(n_out, n_out, 3, 1, 1, bias = False),
            BatchNorm2d(n_out)
        )
        self.shortcut = Sequential()
        if stride != 1 or n_in != n_out:
            self.shortcut = Sequential(
                Conv2d(n_in, n_out, 1, stride, 0, bias = False),
                BatchNorm2d(n_out)
            )
        self.relu = ReLU(inplace = True)

    def forward(self, x):
        return self.relu(self.residual(x) + self.shortcut(x))

def make_resnet_layer(n_blocks, n_in, n_out, stride):
    strides = [stride] + [1] * (n_blocks - 1)
    for stride in strides:
        yield ResNetBasicBlock(n_in, n_out, stride)
        n_in = n_out

class ResNet(Module):
    def __init__(self, layers, n_cls):
        super().__init__()

        # Prelude matches Bu2023's version.
        self.conv1 = Conv2d(3, 64, 3, 1, 1, bias = False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace = True)

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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ResNetSmall(Module):
    def __init__(self, layers, n_cls):
        super().__init__()

        # Prelude matches Bu2023's version.
        self.conv1 = Conv2d(3, 16, 3, 1, 1, bias = False)
        self.bn1 = BatchNorm2d(16)
        self.relu = ReLU(inplace = True)

        # Layers
        self.layer1 = Sequential(*make_resnet_layer(layers[0], 16, 16, 1))
        self.layer2 = Sequential(*make_resnet_layer(layers[1], 16, 32, 2))
        self.layer3 = Sequential(*make_resnet_layer(layers[2], 32, 64, 2))

        self.ap = AdaptiveAvgPool2d((1, 1))
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

########################################################################
# DenseNet 121
########################################################################
GROWTH_RATE = 32
N_BLOCKS = [6, 12, 24, 16]
REDUCTION = 0.5

class Bottleneck(Module):
    def __init__(self, n_in):
        super(Bottleneck, self).__init__()
        self.seq = Sequential(
            BatchNorm2d(n_in),
            ReLU(inplace = True),
            # 1x1 convolution
            Conv2d(n_in, 4 * GROWTH_RATE, 1, bias=False),
            BatchNorm2d(4 * GROWTH_RATE),
            ReLU(inplace = True),
            Conv2d(
                4 * GROWTH_RATE, GROWTH_RATE, 3,
                padding=1, bias=False
            )
        )


    def forward(self, x):
        xp = self.seq(x)
        return torch.cat([xp, x], 1)

def build_dense_layers(n_chans, n_block):
    layers = []
    for i in range(n_block):
        layers.append(Bottleneck(n_chans))
        n_chans += GROWTH_RATE
    return Sequential(*layers)

def build_transition(n_in, n_out):
    return Sequential(
        BatchNorm2d(n_in),
        ReLU(),
        Conv2d(n_in, n_out, 1, bias=False),
        AvgPool2d(2)
    )

class DenseNet(Module):
    def __init__(self, n_cls):
        super(DenseNet, self).__init__()

        n_chans = 2 * GROWTH_RATE
        self.conv1 = Conv2d(3, n_chans, 3, padding=1, bias=False)

        self.dense1 = build_dense_layers(n_chans, N_BLOCKS[0])
        n_chans += N_BLOCKS[0] * GROWTH_RATE
        n_chans_out = int(floor(n_chans*REDUCTION))
        self.trans1 = build_transition(n_chans, n_chans_out)
        n_chans = n_chans_out

        self.dense2 = build_dense_layers(n_chans, N_BLOCKS[1])
        n_chans += N_BLOCKS[1] * GROWTH_RATE
        n_chans_out = int(floor(n_chans*REDUCTION))
        self.trans2 = build_transition(n_chans, n_chans_out)
        n_chans = n_chans_out

        self.dense3 = build_dense_layers(n_chans, N_BLOCKS[2])
        n_chans += N_BLOCKS[2] * GROWTH_RATE
        n_chans_out = int(floor(n_chans*REDUCTION))
        self.trans3 = build_transition(n_chans, n_chans_out)
        n_chans = n_chans_out

        self.dense4 = build_dense_layers(n_chans, N_BLOCKS[3])
        n_chans += N_BLOCKS[3] * GROWTH_RATE

        self.linear = Sequential(
            BatchNorm2d(n_chans),
            ReLU(),
            AvgPool2d(4),
            Flatten(),
            Linear(n_chans, n_cls)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)
        x = self.linear(x)
        return x

########################################################################
# VGG16
########################################################################

# Build the VGG16 layers
def build_vgg_layers(n_cls):
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
    yield Flatten()
    yield Linear(512, 4096)
    yield ReLU(inplace = True)
    yield Linear(4096, 4096)
    yield ReLU(inplace = True)
    yield Linear(4096, n_cls)

class VGG16(Module):
    def __init__(self, n_cls):
        super(VGG16, self).__init__()
        assert n_cls <= 100
        self.features = Sequential(*build_vgg_layers(n_cls))
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
        return self.features(x)

class QCFSNetwork(Module):
    def __init__(self, net, theta, l):
        super().__init__()
        replace_modules(
            net,
            lambda m: isinstance(m, ReLU),
            lambda m: QCFS(theta, l)
        )
        self.n_time_steps = None
        self.net = net

    def set_snn_mode(self, n_time_steps):
        self.n_time_steps = n_time_steps
        for m in self.net.modules():
            if isinstance(m, QCFS):
                m.n_time_steps = n_time_steps

    def forward(self, x):
        if self.n_time_steps > 0:
            for m in self.net.modules():
                m.mem = None
            y = [self.net.forward(x) for _ in range(self.n_time_steps)]
            y = torch.stack(y)
            return y.mean(0)
        return self.net.forward(x)

def load_net(net_name, n_cls):
    if net_name == "densenet":
        return DenseNet(n_cls)
    elif net_name == 'vgg16':
        return VGG16(n_cls)
    elif net_name == 'resnet18':
        return ResNet([2, 2, 2, 2], n_cls)
    elif net_name == 'vgg16qcfs':
        net = VGG16(n_cls)
        replace_modules(
            net,
            lambda m: isinstance(m, Conv2d),
            lambda m: Conv2d(
                m.in_channels, m.out_channels,
                m.kernel_size, m.stride, m.padding,
                bias = True
            )
        )
        replace_modules(
            net,
            lambda m: isinstance(m, MaxPool2d),
            lambda m: AvgPool2d(2)
        )
        return QCFSNetwork(net, 8.0, 8)
    elif net_name == 'resnet18qcfs':
        net = ResNet([2, 2, 2, 2], n_cls)
        return QCFSNetwork(net, 8.0, 8)
    elif net_name == 'resnet20qcfs':
        net = ResNetSmall([3, 3, 3], n_cls)
        return QCFSNetwork(net, 8.0, 8)
    elif net_name == 'resnet34qcfs':
        net = ResNet([3, 4, 6, 3], n_cls)
        return QCFSNetwork(net, 8.0, 8)
    assert False

if __name__ == "__main__":
    from torchinfo import summary
    from torchvision.models import EfficientNet as TVEfficientNet
    from torchvision.models.efficientnet import _efficientnet_conf
    ver = "b0"
    _, res, dropout = PHI_VALUES[ver]
    x = torch.randn((10, 3, res, res))
    net = EfficientNet(ver, 10)
    inv_res_setting, last_channel = \
        _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    print(inv_res_setting)
    net2 = TVEfficientNet(
        num_classes = 10,
        dropout = 0.2,
        inverted_residual_setting = inv_res_setting
    )
    summary(net, input_size = x.shape, device = "cpu")
    summary(net2, input_size = x.shape, device = "cpu")
