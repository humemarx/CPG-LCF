# coding=utf-8
'''
Author: husserl
License: Apache Licence
Software: VSCode
Date: 2023-06-12 08:07:15
LastEditors: husserl
LastEditTime: 2023-06-15 10:25:16
'''
import time
import numpy as np
import math

import torch
from torch import nn
from .. import networks

class CatFPN(nn.Module):
    def __init__(
        self,
        in_channels=[32, 64, 128],
        out_channels=sum([32, 64, 128]),
    ):
        super(CatFPN, self).__init__()
        self.in_channels=in_channels
        self.out_channels=sum(in_channels)

    def forward(self, x_list):
        x_merge = torch.cat(x_list, dim=1)
        return x_merge
    

class SEFPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEFPN, self).__init__()
        cin = sum(in_channels)
        self.sensor_gate = nn.Sequential(
                networks.backbone.conv1x1(cin, cin, bias=True),
                nn.Sigmoid()
            )
        self.point_layer = networks.backbone.conv1x1_bn_relu(cin, out_channels)

    def forward(self, x):
        xi = torch.cat(x, 1)
        gate = self.sensor_gate(xi)
        xo = gate*xi
        xo = self.point_layer(xo)
        return xo
    

class SECONDFPN(nn.Module):
    def __init__(self,                 
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 stage_num=1):
        super(SECONDFPN, self).__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_strides = upsample_strides
        self.stage_num = stage_num

        self.deblocks = nn.ModuleList()
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 :
                upsample_layer = nn.ConvTranspose2d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=False)
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = nn.Conv2d(
                                in_channels=in_channels[i],
                                out_channels=out_channel,
                                kernel_size=stride,
                                stride=stride,
                                bias=False)
            deblock = nn.Sequential(
                        upsample_layer,
                        nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True),
                    )
            self.deblocks.append(deblock)
        print("Finish SECONDFPN Initialization")

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)*self.stage_num
        ups = []
        for i, deblock in enumerate(self.deblocks):
            x_in = []
            for j in range(self.stage_num):
                x_in.append(x[i+j*len(self.in_channels)])
            if len(x_in) > 1:
                x_out = torch.cat(x_in, dim=1)
            else:
                x_out = x_in[0]
            ups.append(deblock(x_out))
                
        # ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return out
