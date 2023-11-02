import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
from typing import Dict, Optional, Tuple, Union
from models.utils.weight_init import (constant_init, kaiming_init)
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

def get_norm_name(norm_type, postfix=''):
    if issubclass(norm_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'+str(postfix)
    elif issubclass(norm_type, _BatchNorm):
        return 'bn'+str(postfix)
    elif issubclass(norm_type, nn.GroupNorm):
        return 'gn'+str(postfix)
    elif issubclass(norm_type, nn.LayerNorm):
        return 'ln'+str(postfix)

class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 conv_type=nn.Conv2d,
                 norm_type=None,
                 act_type=nn.ReLU,
                 inplace=True,
                 order: tuple = ('conv', 'norm', 'act')):
        super().__init__()

        self.order = order
        self.conv_type = conv_type
        self.act_type = act_type
        self.norm_type = norm_type

        self.with_norm = norm_type is not None
        self.with_activation = act_type is not None

        # build convolution layer
        self.conv = conv_type(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

        # build normalization layers
        if self.with_norm:
            norm = norm_type(out_channels)  # type: ignore
            self.norm_name = get_norm_name(norm_type)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None  # type: ignore

        if self.with_activation:
            self.activate = act_type(inplace=inplace)
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and isinstance(self.act_type, nn.LeakyReLU):
                nonlinearity = 'leaky_relu'
                a = 0.01
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.bn, 1, bias=0)

    def forward(self,x):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and self.with_activation:
                x = self.activate(x)
        return x


class conv3x3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, bias=False):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=bias)
    
    def forward(self, x):
        return self.conv(x)


class conv3x3_bn(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1):
        super(conv3x3_bn, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class conv3x3_relu(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1):
        super(conv3x3_relu, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class conv3x3_bn_relu(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1):
        super(conv3x3_bn_relu, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class bn_conv3x3_bn_relu(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1):
        super(bn_conv3x3_bn_relu, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class conv1x1(nn.Module):
    def __init__(self, in_planes, out_planes, bias=False):
        super(conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
    
    def forward(self, x):
        return self.conv(x)


class conv1x1_bn(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(conv1x1_bn, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class conv1x1_relu(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(conv1x1_relu, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class conv1x1_bn_relu(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(conv1x1_bn_relu, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class bn_conv1x1_bn_relu(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(bn_conv1x1_bn_relu, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1


class DeConv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, groups=1, bias=False):
        super(DeConv, self).__init__()
        # kernel_size = stride + 2 * padding
        kernel_size = 4
        padding = None
        if isinstance(stride, int):
            assert stride in [2, 4], "stride must be 2 or 4, but got {}".format(stride)
            padding = (kernel_size - stride) // 2
        elif isinstance(stride, (list, tuple)):
            assert all([x in [1, 2, 4] for x in stride]), "stride must be 1, 2 or 4, but got {}".format(stride)
            padding = [(kernel_size-s) // 2 for s in stride]
        else:
            raise NotImplementedError
        
        self.mod = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
        padding=padding, output_padding=0, groups=groups, bias=bias, dilation=1, padding_mode='zeros')
    
    def forward(self, x):
        return self.mod(x)


class DownSample2D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(DownSample2D, self).__init__()
        self.conv_branch = conv3x3_bn(in_planes, out_planes, stride=stride, dilation=1)
        self.pool_branch = nn.Sequential(
            conv1x1_bn(in_planes, out_planes),
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=1, dilation=1)
        )

        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x_conv = self.conv_branch(x)
        x_pool = self.pool_branch(x)
        x_out = self.act(x_conv + x_pool)
        return x_out


class ChannelAtt(nn.Module):
    def __init__(self, channels, reduction=4):
        super(ChannelAtt, self).__init__()
        self.cnet = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            conv1x1_relu(channels, channels // reduction),
            conv1x1(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        #channel wise
        ca_map = self.cnet(x)
        x = x * ca_map
        return x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, reduction=1, dilation=1, use_att=True):
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential(
            conv3x3_bn_relu(inplanes, inplanes // reduction, stride=1, dilation=1),
            conv3x3_bn(inplanes // reduction, inplanes, stride=1, dilation=dilation)
        )

        self.use_att = use_att
        if self.use_att:
            self.channel_att = ChannelAtt(channels=inplanes, reduction=4)
        
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.layer(x)
        if self.use_att:
            out = self.channel_att(out)
        
        out = self.act(out + x)
        return out


class PredBranch(nn.Module):
    def __init__(self, cin, cout):
        super(PredBranch, self).__init__()
        self.pred_layer = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0, dilation=1)
        )
    
    def forward(self, x):
        pred = self.pred_layer(x)
        return pred