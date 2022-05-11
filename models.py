from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import quantize_to_int

""" Variance Preserving Activations
Gamma from https://github.com/deepmind/deepmind-research/blob/master/nfnets/base.py
"""
class VPActivation(nn.Module):
    def __init__(self, act, gamma):
        super(VPActivation, self).__init__()
        self.act = act
        self.gamma = gamma

    def extra_repr(self):
        return f"(gamma): {self.gamma}"

    def forward(self, x):
        return self.act(x)*self.gamma

acts_dict = {
    'gelu': {'act': nn.GELU(), 'gamma': 1.7015043497085571},
    'leaky_relu': {'act': nn.LeakyReLU(), 'gamma': 1.70590341091156},
    'relu': {'act': nn.ReLU(), 'gamma': 1.7139588594436646},
    'relu6': {'act': nn.ReLU6(), 'gamma': 1.7131484746932983},
    'sigmoid': {'act': nn.Sigmoid(), 'gamma': 4.803835391998291},
    'silu': {'act': nn.SiLU(), 'gamma': 1.7881293296813965},
    'tanh': {'act': nn.Tanh(), 'gamma': 1.5939117670059204},
    None: {'act': nn.Identity(), 'gamma': 1.0},
}
def get_act(name=None, gamma=None):
    act = acts_dict[name.lower()]
    gamma = gamma if (gamma is not None) else act['gamma']
    return VPActivation(act['act'], gamma)

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, se_ratio):
        super(SqueezeExcite, self).__init__()
        mid_channels = quantize_to_int(se_ratio*in_channels, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # The Squeeze
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True),
            get_act('gelu'),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=True),
            get_act('sigmoid', gamma=2.0), # The Excite. 2 rescales the variance back to 1
        )
    
    def forward(self, x):
        return self.se(x)*x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=None, 
                se_ratio=0.0):
        super(ResidualBlock, self).__init__()

        self.pre_activation = nn.Sequential()
        self.pre_activation.add_module("bn", nn.BatchNorm2d(in_channels))
        self.pre_activation.add_module(activation, get_act(activation))

        self.residual = nn.Sequential()
        self.residual.add_module("conv1", nn.Conv2d(in_channels, out_channels,
                kernel_size=(3,3), stride=stride, padding=1, bias=False))
        self.residual.add_module("bn", nn.BatchNorm2d(out_channels))
        self.residual.add_module(activation, get_act(activation))
        self.residual.add_module("conv2", nn.Conv2d(out_channels, out_channels,
                kernel_size=(3,3), stride=1, padding=1, bias=False))
        if se_ratio>0:
            self.residual.add_module("se", SqueezeExcite(out_channels, se_ratio))

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential()
            self.shortcut.add_module("avg", nn.AvgPool2d(kernel_size=stride, stride=stride))
            self.shortcut.add_module("conv", nn.Conv2d(in_channels, out_channels,
                    kernel_size=(1,1), stride=1, bias=False))
        else:
            self.shortcut = None

    def forward(self, x):
        activated = self.pre_activation(x)
        shortcut = x
        if self.shortcut is not None:
            shortcut = self.shortcut(activated)
        residual = self.residual(activated)
        return shortcut + residual

def resnet_block(in_ch, out_ch, blocks, stride, activation=None, se_ratio=None):
    layers = [ResidualBlock(in_ch, out_ch, stride, activation, se_ratio)]
    for i in range(1, blocks):
        layers.append(ResidualBlock(out_ch, out_ch, 1, activation, se_ratio))
    return nn.Sequential(*layers)

class BasicResnet(nn.Sequential):
    def __init__(self, in_channels, filters=[64, 64, 128, 256, 512], blocks=[2, 2, 2, 2], activation=None,
                se_ratio=0.0):
        super(BasicResnet, self).__init__()
        assert (len(filters)-1)==len(blocks), "filters and blocks length do not match."

        self.add_module("conv1", nn.Conv2d(in_channels, filters[0], kernel_size=(5,5),
            stride=1, padding=2, bias=False))
        self.add_module("bn1", nn.BatchNorm2d(filters[0]))

        for idx, num_blocks in enumerate(blocks):
            stride = 1 if idx == 0 else 2  # Downsample after the first block
            self.add_module(f"block{idx}", resnet_block(filters[idx], filters[idx+1], num_blocks, stride, activation, se_ratio))

        self.add_module("avg", nn.AdaptiveAvgPool2d((1,1)))
        self.add_module("flatten", nn.Flatten())


import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import summarize

class LitModel(pl.LightningModule):
    def __init__(self, net, x):
        super().__init__()
        self.net = net
        self.example_input_array = x

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":

    input_size = 32
    in_channels = 3
    filters = [16, 16, 32, 64]
    blocks = [2, 2, 2]
    activation = 'gelu'
    se_ratio = 0

    model = BasicResnet(in_channels, filters, blocks, activation, se_ratio)
    print(model)
    x = torch.ones((1, in_channels, input_size, input_size))
    m = LitModel(model, x)
    print(summarize(m))
