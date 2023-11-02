# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from .decode_head import BaseDecodeHead
from models.utils.weight_init import (constant_init, kaiming_init)
from models.networks.backbone import ConvModule

class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 num_convs=2,
                 out_channels=None,
                 norm_type=None,
                 dropout_ratio=0.1,
                 act_type=nn.ReLU,
                 in_index=-1,
                 ignore_index=255,
                 align_corners=False,
                 input_transform=None,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 enable_fp16=True,
                 use_seg_label=True,
                 pretrained=None):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.use_seg_label = use_seg_label
        super().__init__(in_channels=in_channels, 
                         channels=channels,
                         num_classes=num_classes,
                         out_channels=out_channels,
                         dropout_ratio=dropout_ratio,
                         input_transform=input_transform,
                         norm_type=norm_type,
                         act_type=act_type,
                         in_index=in_index,
                         ignore_index=ignore_index,
                         align_corners=align_corners,
                         enable_fp16 = enable_fp16,
                         pretrained = pretrained)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                norm_type=self.norm_type,
                act_type=self.act_type))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    norm_type=self.norm_type,
                    act_type=self.act_type))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                norm_type=self.norm_type,
                act_type=self.act_type)

        if pretrained is not None:
            self.load_pretrained(pretrained)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        if self.use_seg_label:
            output = self.cls_seg(output)
        return output