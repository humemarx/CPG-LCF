# coding=utf-8
'''
Author: husserl
License: Apache Licence
Software: VSCode
Date: 2023-07-11 09:30:51
LastEditors: husserl
LastEditTime: 2023-07-17 03:09:29
'''
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from models.utils import resize
from .psp_head import PPM
from .decode_head import BaseDecodeHead
from models.utils.weight_init import (constant_init, kaiming_init)
from models.networks.backbone import ConvModule

class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 out_channels=None,
                 norm_type=None,
                 dropout_ratio=0.1,
                 act_type=nn.ReLU,
                 in_index=-1,
                 ignore_index=255,
                 align_corners=False,
                 pool_scales=(1, 2, 3, 6),
                 pretrained = None):
        super().__init__(in_channels=in_channels, 
                         channels=channels,
                         num_classes=num_classes,
                         out_channels=out_channels,
                         dropout_ratio=dropout_ratio,
                         norm_type=norm_type,
                         act_type=act_type,
                         in_index=in_index,
                         ignore_index=ignore_index,
                         align_corners=align_corners,
                         input_transform='multiple_select',
                         pretrained = pretrained)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            norm_type=self.norm_type,
            act_type=self.act_type,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            norm_type=self.norm_type,
            act_type=self.act_type)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                norm_type=self.norm_type,
                act_type=self.act_type,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_type=self.norm_type,
                act_type=self.act_type,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            norm_type=self.norm_type,
            act_type=self.act_type)
        
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output