# Copyright (C) 2024-2025 Björn A. Lindqvist <bjourne@gmail.com>
from math import ceil, floor, sqrt
from torch.autograd import Function
from torch.nn import *
from torch.nn.init import constant_, kaiming_normal_, normal_, uniform_
from torch.nn.functional import relu
from torchvision.transforms import Resize

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
    def __init__(self, theta, l, n_time_steps, spike_prop):
        super().__init__()
        self.theta = Parameter(torch.tensor([theta]), requires_grad=True)
        self.l = l
        self.n_time_steps = n_time_steps
        self.spike_prop = spike_prop

    def forward_qcfs(self, x):
        x = x / self.theta
        x = torch.clamp(x, 0, 1)
        x = GradFloor.apply(x * self.l + 0.5) / self.l
        return x * self.theta

    def forward_if_lbl(self, x):
        y_shape = [self.n_time_steps, x.shape[0] // self.n_time_steps]
        y_shape += x.shape[1:]
        x = x.view(y_shape)
        theta = self.theta.data

        mem = 0.5 * theta
        y = torch.empty(y_shape)
        for t in range(self.n_time_steps):
            mem = mem + x[t, ...]
            spike = (mem - theta >= 0).float() * theta
            mem -= spike
            y[t] = spike
        y = y.flatten(0, 1).contiguous()
        return y

    def forward_if_tbt(self, x):
        theta = self.theta.data
        if self.mem is None:
            self.mem = torch.zeros_like(x) + theta * 0.5
        self.mem += x
        x = (self.mem - theta >= 0).float() * theta
        self.mem -= x
        return x

    def forward(self, x):
        if self.n_time_steps > 0:
            if self.spike_prop == "lbl":
                return self.forward_if_lbl(x)
            return self.forward_if_tbt(x)
        return self.forward_qcfs(x)

def replace_modules(mod, match_fun, new_fun):
    for name, submod in mod.named_children():
        if match_fun(submod):
            setattr(mod, name, new_fun(submod))
        replace_modules(submod, match_fun, new_fun)

def is_relu(mod):
    return isinstance(mod, ReLU)

########################################################################
# AlexNet
########################################################################
def alex_net(n_cls):
    layers = [
        # Ensure input is big enough
        Resize((227, 227), antialias = True),
        Conv2d(3, 64, 11, 4, 2),
        ReLU(inplace = True),
        MaxPool2d(3, 2),
        Conv2d(64, 192, 5, 1, 2),
        ReLU(inplace = True),
        MaxPool2d(3, 2),
        Conv2d(192, 384, 3, 1, 1),
        ReLU(inplace = True),
        Conv2d(384, 256, 3, 1, 1),
        ReLU(inplace = True),
        Conv2d(256, 256, 3, 1, 1),
        ReLU(inplace = True),
        MaxPool2d(3, 2),
        AdaptiveAvgPool2d(6),
        Flatten(),
        Linear(256 * 6 * 6, 4096),
        ReLU(inplace = True),
        Dropout(0.5),
        Linear(4096, 4096),
        ReLU(inplace = True),
        Dropout(0.5),
        Linear(4096, n_cls)
    ]
    return Sequential(*layers)


########################################################################
# EfficientNet
########################################################################
EFF_BASE = [
    # expand, n_in, repeats, stride, k_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]
EFF_SD_PROB = 0.2

# dropout, width mult, depth mult, (bn_eps, bn_mom)
EFF_CONFIGS = {
    "b0" : (0.2, 1.0, 1.0, (1e-5, 0.10)),
    "b1" : (0.2, 1.0, 1.1, (1e-5, 0.10)),
    "b2" : (0.3, 1.1, 1.2, (1e-5, 0.10)),
    "b3" : (0.3, 1.2, 1.4, (1e-5, 0.10)),
    "b4" : (0.4, 1.4, 1.8, (1e-5, 0.10)),
    "b5" : (0.4, 1.6, 2.2, (1e-3, 0.01)),
    "b6" : (0.5, 1.8, 2.6, (1e-3, 0.01)),
    "b7" : (0.5, 2.0, 3.1, (1e-3, 0.01))
}

def round_align(v, div):
    """
    I found this algo in torchvision's source code.
    """
    min_v = div
    new_v = max(min_v, int(v + div / 2) // div * div)
    if new_v < 0.9 * v:
        new_v += div
    return new_v

class Conv2dBNAct(Module):
    def __init__(
        self,
        n_in, n_out,
        k_size, stride, padding, groups,
        bn_args
    ):
        super().__init__()
        self.conv = Conv2d(
            n_in, n_out, k_size,
            stride, padding,
            groups = groups,
            bias = False
        )
        eps, momentum = bn_args
        self.bn = BatchNorm2d(n_out, eps = eps, momentum = momentum)
        self.silu = SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x

# Attention scores for each of the channels
class SqueezeExcitation(Module):
    def __init__(self, n_in, n_squeezed):
        super(SqueezeExcitation, self).__init__()
        self.se = Sequential(
            AdaptiveAvgPool2d(1),
            Conv2d(n_in, n_squeezed, 1),
            SiLU(),
            Conv2d(n_squeezed, n_in, 1),
            Sigmoid()
        )

    def forward(self, x):
        # How can this be element-wise?
        return x * self.se(x)

class MBConv(Module):
    def __init__(
        self,
        n_in, n_out,
        k_size, stride, kill_p, exp_ratio, bn_args
    ):
        super().__init__()

        n_hidden = round_align(n_in * exp_ratio, 8)
        layers = []
        self.kill_p = kill_p
        self.use_residual = n_in == n_out and stride == 1
        self.expand_conv = None
        if n_in != n_hidden:
            layers.append(
                Conv2dBNAct(
                    n_in, n_hidden,
                    1, 1, 0, 1,
                    bn_args
                )
            )

        n_squeezed = max(1, n_in // 4)
        layers.extend([
            # Depthwise conv
            Conv2dBNAct(
                n_hidden, n_hidden,
                k_size, stride, k_size // 2, n_hidden,
                bn_args
            ),
            # Sequeeze
            SqueezeExcitation(n_hidden, n_squeezed),
            # Project
            Conv2d(n_hidden, n_out, 1, bias = False),
            BatchNorm2d(n_out, eps = bn_args[0], momentum = bn_args[1])
        ])
        self.block = Sequential(*layers)

    def stochastic_depth(self, x):
        if not self.training:
            return x
        survival_p = 1 - self.kill_p
        size = [x.shape[0]] + [1] * (x.ndim - 1)
        noise = torch.empty(size, dtype = x.dtype, device = x.device)
        noise = noise.bernoulli_(survival_p)
        if survival_p > 0.0:
            noise.div_(survival_p)
        return x * noise

    def forward(self, x):
        xp = self.block(x)
        if self.use_residual:
            xp = self.stochastic_depth(xp) + x
        return xp

def make_effnet_layers(width_mult, depth_mult, n_last, bn_args):
    n_in = round_align(32 * width_mult, 8)
    yield Resize((224, 224), antialias = True)
    yield Conv2dBNAct(3, n_in, 3, 2, 1, 1, bn_args)

    mbconvs_per_stage = [ceil(n_reps * depth_mult)
                         for (_, _, n_reps, _, _) in EFF_BASE]
    n_tot_mbconvs = sum(mbconvs_per_stage)
    at = 0
    for n_mbconvs, stage_data in zip(mbconvs_per_stage, EFF_BASE):
        exp_ratio, n_chans, _, stride, k_size = stage_data
        n_out = round_align(n_chans * width_mult, 8)
        for i in range(n_mbconvs):
            kill_p = EFF_SD_PROB * at / n_tot_mbconvs
            yield MBConv(
                n_in, n_out,
                k_size,
                stride if i == 0 else 1,
                kill_p,
                exp_ratio,
                bn_args
            )
            at += 1
            n_in = n_out
    yield Conv2dBNAct(n_in, n_last, 1, 1, 0, 1, bn_args)


class EfficientNet(Module):
    def __init__(self, ver, n_classes):
        super().__init__()

        # Width, depth, and dropout
        dropout, width_mult, depth_mult, bn_args = EFF_CONFIGS[ver]
        n_last = ceil(1280 * width_mult)
        self.pool = AdaptiveAvgPool2d(1)

        self.features = Sequential(
            *make_effnet_layers(width_mult, depth_mult, n_last, bn_args)
        )
        self.classifier = Sequential(
            Flatten(),
            Dropout(dropout),
            Linear(n_last, n_classes)
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

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)



########################################################################
# ResNets
########################################################################

# Pre-activated resnet block. Follows implementation:
#
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
class ResNetPABlock(Module):
    def __init__(self, n_in, n_out, stride):
        super().__init__()

        self.preact = Sequential(
            BatchNorm2d(n_in),
            ReLU(inplace = True),
        )
        self.shortcut = Sequential()
        if stride != 1 or n_in != n_out:
            # Unclear whether there should be bias/batch norm in the
            # shortcut or not.
            self.shortcut = Sequential(
                Conv2d(n_in, n_out, 1, stride, bias = False),
                BatchNorm2d(n_out)
            )
        self.residual = Sequential(
            Conv2d(n_in, n_out, 3, stride, 1, bias = False),
            BatchNorm2d(n_out),
            ReLU(inplace = True),
            Conv2d(n_out, n_out, 3, 1, 1, bias = False)
        )

    def forward(self, x):
        xp = self.preact(x)
        return self.residual(xp) + self.shortcut(xp)


# No expansion factor yet
class ResNetBlock(Module):
    def __init__(self, n_in, n_out, stride):
        super().__init__()

        # Downsampling in the first convolution
        self.residual = Sequential(
            Conv2d(n_in, n_out, 3, stride, 1, bias = False),
            BatchNorm2d(n_out),
            ReLU(inplace = True),
            Conv2d(n_out, n_out, 3, 1, 1, bias = False),
            BatchNorm2d(n_out)
        )
        self.shortcut = Sequential()
        if stride != 1 or n_in != n_out:
            # Some implementations use a biased conv2d for the
            # downsampling instead of batch norm. It shouldn't matter
            # much.
            self.shortcut = Sequential(
                Conv2d(n_in, n_out, 1, stride, 0, bias = False),
                BatchNorm2d(n_out)
            )
        self.relu = ReLU(inplace = True)

    def forward(self, x):
        return self.relu(self.residual(x) + self.shortcut(x))

def make_resnet_blocks(is_pa, layers, n_in, n_cls):
    block_cls = ResNetPABlock if is_pa else ResNetBlock
    for n_blocks, n_out, stride in layers:
        strides = [stride] + [1] * (n_blocks - 1)
        for stride in strides:
            yield block_cls(n_in, n_out, stride)
            n_in = n_out

    # The pre-activated resnet blocks does not finish with batch
    # norm/relu, so we need it here (I think).
    if is_pa:
        yield BatchNorm2d(n_in)
        yield ReLU(inplace = True)

    yield AdaptiveAvgPool2d((1, 1))
    yield Flatten()
    yield Linear(n_in, n_cls)

class ResNet(Module):
    def __init__(self, layers, n_in, n_cls):
        super().__init__()

        # Prelude matches Bu2023's version.
        self.conv1 = Conv2d(3, n_in, 3, 1, 1, bias = False)
        self.bn1 = BatchNorm2d(n_in)
        self.relu = ReLU(inplace = True)

        # Layers
        self.blocks = Sequential(
            *make_resnet_blocks(False, layers, n_in, n_cls)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.blocks(x)

class ResNetPA(Module):
    def __init__(self, layers, n_in, n_cls):
        super().__init__()
        self.conv1 = Conv2d(3, n_in, 3, 1, 1, bias = False)

        # Since it is preact, we don't need conv or bn before the
        # blocks.
        self.blocks = Sequential(
            *make_resnet_blocks(True, layers, n_in, n_cls)
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.blocks(x)


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
    vgg16_layers = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M",
    ]
    n_chans_in = 3
    for v in vgg16_layers:
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

def vgg16(n_cls):
    net = Sequential(*build_vgg_layers(n_cls))
    for m in net.modules():
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
    return net

class QCFSNetwork(Module):
    def __init__(self, net, theta, l, n_time_steps, spike_prop):
        super().__init__()
        replace_modules(
            net,
            lambda m: isinstance(m, ReLU),
            lambda m: QCFS(theta, l, n_time_steps, spike_prop)
        )
        self.n_time_steps = n_time_steps
        self.net = net
        self.spike_prop = spike_prop

    def forward_lbl(self, x):
        bs, *rem = x.shape
        x = x.unsqueeze(1).repeat(self.n_time_steps, 1, 1, 1, 1)
        x = x.flatten(0, 1).contiguous()

        x = self.net.forward(x)
        _, n_cls = x.shape
        x = x.view((self.n_time_steps, bs, n_cls))
        return x.mean(0)

    def forward_tbt(self, x):
        for m in self.net.modules():
            m.mem = None
        y = [self.net.forward(x) for _ in range(self.n_time_steps)]
        y = torch.stack(y)
        return y.mean(0)

    def forward(self, x):
        if self.n_time_steps > 0:
            if self.spike_prop == "lbl":
                return self.forward_lbl(x)
            return self.forward_tbt(x)
        return self.net.forward(x)

def load_net(net_name, n_cls, n_time_steps, spike_prop):
    if net_name == "alexnet":
        return alex_net(n_cls)
    elif net_name == "densenet":
        return DenseNet(n_cls)
    elif net_name == "efficientnet-b0":
        return EfficientNet("b0", n_cls)
    elif net_name == 'resnet18':
        return ResNet(
            [(2, 64, 1), (2, 128, 2), (2, 256, 2), (2, 512, 2)],
            64,
            n_cls
        )
    elif net_name == 'resnet18qcfs':
        net = ResNet([2, 2, 2, 2], n_cls)
        return QCFSNetwork(net, 8.0, 8, n_time_steps)
    elif net_name == 'resnet20':
        return ResNet([(3, 16, 1), (3, 32, 2), (3, 64, 2)], 16, n_cls)
    elif net_name == "resnet20-pa":
        return ResNetPA([(3, 16, 1), (3, 32, 2), (3, 64, 2)], 16, n_cls)
    elif net_name == 'resnet20qcfs':
        net = ResNetSmall([3, 3, 3], n_cls)
        return QCFSNetwork(net, 8.0, 8, n_time_steps)
    elif net_name == 'resnet34qcfs':
        net = ResNet([3, 4, 6, 3], n_cls)
        return QCFSNetwork(net, 8.0, 8 , n_time_steps)
    elif net_name == 'vgg16':
        return vgg16(n_cls)
    elif net_name == 'vgg16qcfs':
        net = vgg16(n_cls)
        # Shouldn't the batch norms be removed??
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
        return QCFSNetwork(net, 8.0, 8, n_time_steps, spike_prop)
    assert False

def cnt_params(net):
    import numpy as np
    params = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in params])
    return params

def sum_attr(net, cls, attr):
    from numbers import Number
    tot = 0
    for m in net.modules():
        if isinstance(m, cls):
            o = getattr(m, attr)
            if not isinstance(o, Number):
                o = sum(o)
            tot += o
    return tot

def tests():
    from torchinfo import summary
    from torchvision.ops import StochasticDepth

    b0 = EfficientNet("b0", 10)

    assert cnt_params(b0) == 4020358
    assert cnt_params(EfficientNet("b1", 10)) == 6525994
    assert cnt_params(EfficientNet("b2", 10)) == 7715084
    assert cnt_params(EfficientNet("b3", 10)) == 10711602
    assert cnt_params(EfficientNet("b4", 10)) == 17566546
    assert cnt_params(EfficientNet("b5", 10)) == 28361274
    assert cnt_params(EfficientNet("b6", 10)) == 40758754

    b7 = EfficientNet("b7", 10)
    assert cnt_params(b7) == 63812570
    assert round(sum_attr(b7, BatchNorm2d, "eps"), 3) == 0.163
    assert sum_attr(b7, Conv2d, "kernel_size") == 890
    assert round(sum_attr(b7, MBConv, "kill_p"), 2) == 5.4

    net = load_net("resnet20", 100, 0)
    assert cnt_params(net) == 278324

    net = load_net("resnet20-pa", 100, 0)
    assert cnt_params(net) == 278324

    net = load_net("resnet18", 100, 0)
    assert cnt_params(net) == 11220132

if __name__ == "__main__":
    #tests()
    from torchinfo import summary
    from calflops import calculate_flops

    net = AlexNet(1000)
    shape = 1, 3, 227, 227
    summary(net, input_size = shape, device = "cpu")



    flops, macs, params = calculate_flops(
        model=net,
        input_shape=shape,
        output_as_string=True,
        output_precision=4
    )
    print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
