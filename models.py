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

""" Anti aliasing CNNs
https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
The code is inspired by the link above.
"""
def make_gaussian_kernel2d(kernel=3, std=1):
    """Returns a 2D Gaussian kernel array."""
    n = torch.arange(0,kernel)-(kernel-1.0)/2.0
    gk1d = torch.exp(-n**2/(2*std*std))
    gk2d = torch.outer(gk1d, gk1d)
    gk2d = gk2d / torch.sum(gk2d)
    return gk2d

class CustomBlurPool(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=0):
        super(CustomBlurPool, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding
        kernel = make_gaussian_kernel2d(kernel=kernel_size, std=1)
        self.register_buffer('kernel', kernel[None,None,:,:].repeat((channels,1,1,1)))

    def forward(self, x):
        padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        return F.conv2d(padded, self.kernel, None, self.stride, groups=self.channels)

def get_downsample(downsample, stride, channels):
    if downsample=='avg':
        return nn.AvgPool2d(kernel_size=stride, stride=stride)
    elif downsample=='max':
        return nn.MaxPool2d(kernel_size=stride, stride=stride)
    elif downsample=='blur':
        return CustomBlurPool(channels, kernel_size=3, stride=stride, padding=1)

class StochasticDepth(nn.Module):
    def __init__(self, drop_rate=0.0):
        super(StochasticDepth, self).__init__()
        self.drop_rate = drop_rate

    def extra_repr(self):
        return f"drop_rate={self.drop_rate}"

    def forward(self, x):
        if not self.training or self.drop_rate==0.0:
            return x
        rand_tensor = torch.rand(x.shape[0], 1, 1, 1).type_as(x).to(x.device)
        binary_tensor = torch.floor(rand_tensor+1-self.drop_rate)
        return x*binary_tensor

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
                downsample='avg', bottleneck_ratio=0, se_ratio=0, stochastic_depth=0):
        super(ResidualBlock, self).__init__()

        self.pre_activation = nn.Sequential()
        self.pre_activation.add_module("bn", nn.BatchNorm2d(in_channels))
        self.pre_activation.add_module(activation, get_act(activation))

        self.residual = nn.Sequential()
        if bottleneck_ratio>0:
            mid_channels = quantize_to_int(out_channels*bottleneck_ratio, 8)
            self.residual.add_module("conv1", nn.Conv2d(in_channels, mid_channels,
                    kernel_size=(1,1), stride=1, padding=0, bias=False))
            self.residual.add_module("bn1", nn.BatchNorm2d(mid_channels))
            self.residual.add_module(activation+"1", get_act(activation))
            self.residual.add_module("conv2", nn.Conv2d(mid_channels, mid_channels,
                    kernel_size=(3,3), stride=stride, padding=1, bias=False))
            self.residual.add_module("bn2", nn.BatchNorm2d(mid_channels))
            self.residual.add_module(activation+"2", get_act(activation))
            self.residual.add_module("conv3", nn.Conv2d(mid_channels, out_channels,
                    kernel_size=(1,1), stride=1, padding=0, bias=False))
        else:
            self.residual.add_module("conv1", nn.Conv2d(in_channels, out_channels,
                    kernel_size=(3,3), stride=stride, padding=1, bias=False))
            self.residual.add_module("bn", nn.BatchNorm2d(out_channels))
            self.residual.add_module(activation, get_act(activation))
            self.residual.add_module("conv2", nn.Conv2d(out_channels, out_channels,
                    kernel_size=(3,3), stride=1, padding=1, bias=False))
        if se_ratio>0:
            self.residual.add_module("se", SqueezeExcite(out_channels, se_ratio))
        if stochastic_depth>0:
            self.residual.add_module("stochastic_depth", StochasticDepth(stochastic_depth))
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential()
            self.shortcut.add_module(downsample, get_downsample(downsample, stride, in_channels))
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

def resnet_stage(in_ch, out_ch, blocks, stride, activation=None,
                downsample='avg', bottleneck_ratio=0, se_ratio=0, stochastic_depth=0):
    layers = [ResidualBlock(in_ch, out_ch, stride, activation, downsample, bottleneck_ratio, se_ratio, stochastic_depth=0)]
    for i in range(1, blocks):
        drop_rate = stochastic_depth * (1+i)/blocks  # Linearly increase drop rate over a stage
        layers.append(ResidualBlock(out_ch, out_ch, 1, activation, downsample, bottleneck_ratio, se_ratio, drop_rate))
    return nn.Sequential(*layers)

class BasicResnet(nn.Sequential):
    def __init__(self, in_channels, filters=[64, 64, 128, 256, 512], blocks=[2, 2, 2, 2], activation=None,
                downsample='avg', bottleneck_ratio=0.0, se_ratio=0.0, stochastic_depth=0.0):
        super(BasicResnet, self).__init__()
        assert (len(filters)-1)==len(blocks), "filters and blocks length do not match."

        self.add_module("conv1", nn.Conv2d(in_channels, filters[0], kernel_size=(5,5),
            stride=1, padding=2, bias=False))
        self.add_module("bn1", nn.BatchNorm2d(filters[0]))

        for idx, num_blocks in enumerate(blocks):
            stride = 1 if idx == 0 else 2  # Downsample after the first block
            self.add_module(f"block{idx}", resnet_stage(filters[idx], filters[idx+1], num_blocks, stride, activation,
                                            downsample, bottleneck_ratio, se_ratio, stochastic_depth))

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

    # Classify
    input_size = 32
    in_channels = 3
    filters = [16, 16, 32, 64]
    blocks = [2, 2, 2]
    activation = 'gelu'
    downsample = 'avg'  # max, avg, blur
    bottleneck_ratio = 0.5
    se_ratio = 0
    stochastic_depth = 0
    model = BasicResnet(in_channels, filters, blocks, activation, downsample, bottleneck_ratio, se_ratio, stochastic_depth)
    print(model)
    x = torch.ones((1, in_channels, input_size, input_size))
    m = LitModel(model, x)
    print(summarize(m))

    # Segment
    input_size = 32
    in_channels = 3
    filters = [16, 16, 32, 64]
    activation = 'gelu'
    downsample = 'avg'  # max, avg, blur

