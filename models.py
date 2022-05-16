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
    'leaky_relu': {'act': nn.LeakyReLU(inplace=True), 'gamma': 1.70590341091156},
    'relu': {'act': nn.ReLU(inplace=True), 'gamma': 1.7139588594436646},
    'relu6': {'act': nn.ReLU6(inplace=True), 'gamma': 1.7131484746932983},
    'sigmoid': {'act': nn.Sigmoid(), 'gamma': 4.803835391998291},
    'silu': {'act': nn.SiLU(inplace=True), 'gamma': 1.7881293296813965},
    'tanh': {'act': nn.Tanh(), 'gamma': 1.5939117670059204},
}
def get_act(name=None, gamma=None):
    if name is None: return nn.Identity()
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
    def __init__(self, in_channels, se_ratio, conv=nn.Conv2d):
        super(SqueezeExcite, self).__init__()
        mid_channels = quantize_to_int(se_ratio*in_channels, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # The Squeeze
            conv(in_channels, mid_channels, kernel_size=1, bias=True),
            get_act('gelu'),
            conv(mid_channels, in_channels, kernel_size=1, bias=True),
            get_act('sigmoid', gamma=2.0), # The Excite. 2 rescales the variance back to 1
        )
    
    def forward(self, x):
        return self.se(x)*x

class WSConv2d(nn.Conv2d):
    """ Weight Standarization Conv2d """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                dilation=1, groups=1, bias=False, eps=1e-4):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                            padding, dilation, groups, bias)
        # Gain is implemented as a learnable a parameter for each output channel
        self.gain = nn.Parameter(torch.ones(self.out_channels))
        # Appendix D calculates the fan-in differently
        self.fan_in = torch.tensor(self.weight[0].numel(), requires_grad=False).type_as(self.weight)
        self.scale = 1/self.fan_in
        # Epsilon, a small constant to avoid dividing by zero
        self.eps = torch.tensor(eps, requires_grad=False).type_as(self.weight)
        self.saved = False  # save copy of normalized weights for inference
        nn.init.xavier_normal_(self.weight)

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.weight.requires_grad = mode
        if not mode:
            # When setting to eval, load the standarized weights once and save them
            with torch.set_grad_enabled(mode):
                self.weight.copy_(self.get_standarized_weights().clone().detach().requires_grad_(True))
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def get_standarized_weights(self):
        if self.training:
            weight = F.batch_norm(
                self.weight.reshape(1, self.out_channels, -1),  # input, normalize by channel
                None, None,  # running_mean and running_var are optional
                (self.scale*self.gain),  # weight, scale parameter in regular bn
                None,  # bias optional
                True,  # training, uses the "mini-batch" to normalize, this is simply the conv weights
                0.0,  # momentum
                self.eps,  # eps
            ).reshape_as(self.weight)
        else:
            weight = self.weight
        return weight

    def forward(self, x):
        weight = self.get_standarized_weights()
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=None, 
                downsample='avg', bottleneck_ratio=0, se_ratio=0, stochastic_depth=0,
                conv=nn.Conv2d, norm=nn.BatchNorm2d, alpha=1.0, beta=1.0):
        super(ResidualBlock, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.pre_activation = nn.Sequential()
        if norm: self.pre_activation.add_module("norm", norm(in_channels))
        self.pre_activation.add_module(activation, get_act(activation))

        self.residual = nn.Sequential()
        if bottleneck_ratio>0:
            mid_channels = quantize_to_int(out_channels*bottleneck_ratio, 8)
            self.residual.add_module("conv1", conv(in_channels, mid_channels,
                    kernel_size=(1,1), stride=1, padding=0, bias=False))
            if norm: self.residual.add_module("norm1", norm(mid_channels))
            self.residual.add_module(activation+"1", get_act(activation))
            self.residual.add_module("conv2", conv(mid_channels, mid_channels,
                    kernel_size=(3,3), stride=stride, padding=1, bias=False))
            if norm: self.residual.add_module("norm2", norm(mid_channels))
            self.residual.add_module(activation+"2", get_act(activation))
            self.residual.add_module("conv3", conv(mid_channels, out_channels,
                    kernel_size=(1,1), stride=1, padding=0, bias=False))
        else:
            self.residual.add_module("conv1", conv(in_channels, out_channels,
                    kernel_size=(3,3), stride=stride, padding=1, bias=False))
            if norm: self.residual.add_module("norm1", norm(out_channels))
            self.residual.add_module(activation, get_act(activation))
            self.residual.add_module("conv2", conv(out_channels, out_channels,
                    kernel_size=(3,3), stride=1, padding=1, bias=False))
        if se_ratio>0:
            self.residual.add_module("se", SqueezeExcite(out_channels, se_ratio, conv=conv))
        if stochastic_depth>0:
            self.residual.add_module("stochastic_depth", StochasticDepth(stochastic_depth))
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential()
            self.shortcut.add_module(downsample, get_downsample(downsample, stride, in_channels))
            self.shortcut.add_module("conv", conv(in_channels, out_channels,
                    kernel_size=(1,1), stride=1, bias=False))
        else:
            self.shortcut = None

    def forward(self, x):
        activated = self.pre_activation(x)*self.beta
        shortcut = x
        if self.shortcut is not None:
            shortcut = self.shortcut(activated)
        residual = self.residual(activated)
        return shortcut + residual*self.alpha

def resnet_stage(in_ch, out_ch, blocks, stride, activation=None,
                downsample='avg', bottleneck_ratio=0, se_ratio=0, stochastic_depth=0,
                conv=nn.Conv2d, norm=nn.BatchNorm2d, alpha=1.0):
    layers = [ResidualBlock(in_ch, out_ch, stride, activation, downsample, bottleneck_ratio, se_ratio, 0, conv, norm)]
    for i in range(1, blocks):
        beta = (i*alpha**2)**-0.5 if alpha!=1.0 else 1.0
        drop_rate = stochastic_depth * (1+i)/blocks  # Linearly increase drop rate over a stage
        layers.append(ResidualBlock(out_ch, out_ch, 1, activation,
            downsample, bottleneck_ratio, se_ratio, drop_rate,
            conv, norm, alpha, beta))
    return nn.Sequential(*layers)

class BasicResnet(nn.Sequential):
    def __init__(self, in_channels, filters=[64, 64, 128, 256, 512], blocks=[2, 2, 2, 2], activation=None,
                downsample='avg', bottleneck_ratio=0.0, se_ratio=0.0, stochastic_depth=0.0,
                conv=nn.Conv2d, norm=nn.BatchNorm2d, alpha=1.0):
        super(BasicResnet, self).__init__()
        assert (len(filters)-1)==len(blocks), "filters and blocks length do not match."

        self.add_module("conv1", conv(in_channels, filters[0], kernel_size=(5,5),
            stride=1, padding=2, bias=False))
        if norm: self.add_module("norm1", norm(filters[0]))

        for idx, num_blocks in enumerate(blocks):
            stride = 1 if idx == 0 else 2  # Downsample after the first block
            self.add_module(f"block{idx}", resnet_stage(filters[idx], filters[idx+1], num_blocks, stride, activation,
                                            downsample, bottleneck_ratio, se_ratio, stochastic_depth, conv, norm, alpha))

        self.add_module("avg", nn.AdaptiveAvgPool2d((1,1)))
        self.add_module("flatten", nn.Flatten())

@torch.jit.script
def autocrop(encoder_features: torch.Tensor, decoder_features: torch.Tensor):
    """ Center crop the encoder down to the size of the decoder """
    if encoder_features.shape[2:] != decoder_features.shape[2:]:
        ds = encoder_features.shape[2:]
        es = decoder_features.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        encoder_features = encoder_features[:, :,
                        ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                        ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                        ]
    return encoder_features, decoder_features

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                stride=1, padding=1, activation=None):
        super().__init__()
        self.add_module("conv1", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.add_module("bn1", nn.BatchNorm2d(out_channels))
        self.add_module("act1", get_act(activation))
        self.add_module("conv2", nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        self.add_module("bn2", nn.BatchNorm2d(out_channels))
        self.add_module("act2", get_act(activation))

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                stride=1, padding=1, activation=None, downsample='avg'):
        super().__init__()
        self.add_module(downsample, get_downsample(downsample, stride, in_channels))
        self.add_module("double_conv", DoubleConv(in_channels, out_channels, kernel_size, 1, padding, activation))

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(2*out_channels, out_channels, activation=activation)

    def forward(self, encoder_features: torch.Tensor, decoder_features: torch.Tensor):
        decoder_features = self.up(decoder_features)
        encoder_features, decoder_features = autocrop(encoder_features, decoder_features)
        x = torch.cat([encoder_features, decoder_features], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, filters=[8, 16, 32, 64], activation=None,
                downsample='avg'):
        super(UNet, self).__init__()
        self.input = DoubleConv(in_channels, filters[0])
        self.encoders = nn.ModuleList([Down(filters[i], filters[i+1], stride=2, activation=activation, downsample=downsample) for i in range(len(filters)-1)])
        self.decoders = nn.ModuleList([Up(filters[i+1], filters[i], activation=activation) for i in reversed(range(len(filters)-1))])
        self.out = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        features = [self.input(x)]
        for level, enc in enumerate(self.encoders):
            features.append(enc(features[level]))
        up = features[len(self.decoders)]
        for level, dec in enumerate(self.decoders):
            up = dec(features[len(self.decoders)-level-1], up)
        return self.out(up)

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
    filters = [32, 32, 64, 128]
    blocks = [2, 2, 2]
    activation = 'gelu'  # gelu, leaky_relu, relu, relu6, sigmoid, silu, tanh
    downsample = 'avg'  # max, avg, blur
    bottleneck_ratio = 0
    se_ratio = 0
    stochastic_depth = 0
    conv = WSConv2d  # None, nn.Conv2d, WSConv2d
    norm = None # None, nn.BatchNorm2d
    alpha = 0.2
    x = torch.ones((1, in_channels, input_size, input_size))
    resnet = BasicResnet(in_channels, filters, blocks, activation,
                downsample, bottleneck_ratio, se_ratio, stochastic_depth,
                conv, norm, alpha)
    print(resnet)
    m = LitModel(resnet, x)
    print(summarize(m))

    # Segment
    # input_size = 256
    # in_channels = 3
    # out_channels = 1
    # filters = [16, 16, 32, 64]
    # activation = 'gelu'  # gelu, leaky_relu, relu, relu6, sigmoid, silu, tanh
    # downsample = 'avg'  # max, avg, blur
    # inp = torch.rand(1, in_channels, input_size, input_size)
    # unet = UNet(in_channels, out_channels, filters, activation, downsample)
    # print(unet)
    # m = LitModel(unet, inp)
    # print(summarize(m))
