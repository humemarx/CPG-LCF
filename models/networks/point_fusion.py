import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbone

from torchvision.ops import deform_conv2d
try:
    from ops_libs.dcn import DeformConv
except:
    print("Deformable Convolution not built!")

import pdb
import numpy as np


class CatFusion(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(CatFusion, self).__init__()
        self.in_channel_list = in_channel_list
        self.out_channel = out_channel

        assert len(self.in_channel_list) >= 2

        s = 0
        for in_channel in self.in_channel_list:
            s = s + in_channel
        
        c_mid = max(s // 2, out_channel)
        self.merge_layer = nn.Sequential(
            backbone.conv1x1_bn_relu(s, c_mid),
            backbone.conv1x1_bn_relu(c_mid, out_channel)
        )
    
    def forward(self, x_list):
        #pdb.set_trace()
        x_merge = torch.cat(x_list, dim=1)
        x_out = self.merge_layer(x_merge)
        return x_out
    
class SPCatFusion(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(SPCatFusion, self).__init__()
        self.in_channel_list = in_channel_list
        self.out_channel = out_channel

        assert len(self.in_channel_list) >= 2

        s = 0
        for in_channel in self.in_channel_list:
            s = s + in_channel
        
        c_mid = max(s // 2, out_channel)
        self.merge_layer = nn.Sequential(
            backbone.conv3x3_bn_relu(s, c_mid),
            backbone.conv3x3_bn_relu(c_mid, out_channel)
        )
    
    def forward(self, x_list):
        #pdb.set_trace()
        x_merge = torch.cat(x_list, dim=1)
        x_out = self.merge_layer(x_merge)
        return x_out

class SPContextFusion(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(SPContextFusion, self).__init__()
        cin = sum(in_channel_list)
        self.sensor_gate = nn.Sequential(
                backbone.conv3x3(cin, cin, stride=1, dilation=1, bias=True),
                nn.Sigmoid()
            )
        self.point_layer = backbone.conv3x3_bn_relu(cin, out_channel)

    def forward(self, x):
        xi = torch.cat(x, 1)
        gate = self.sensor_gate(xi)
        xo = gate*xi
        xo = self.point_layer(xo)
        return xo


class SPDCNFusion(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(SPDCNFusion, self).__init__()
        cin = sum(in_channel_list)
        kernel_size = 3
        deformable_groups = 1
        offset_channels = kernel_size * kernel_size * 2

        self.conv_offset = nn.Conv2d(cin, deformable_groups * offset_channels, 1, bias=True)
        self.conv_mask = nn.Conv2d(cin, kernel_size*kernel_size, 1, bias=True)
        self.conv = nn.Conv2d(cin, out_channel, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        nn.init.constant_(self.conv_mask.weight, 0.5)

    def forward(self, x):
        #pdb.set_trace()
        xi = torch.cat(x, 1)
        offset = self.conv_offset(xi)
        mask = torch.sigmoid(self.conv_mask(xi))
        x_merge = deform_conv2d(xi, offset, weight=self.conv.weight, mask=mask, padding=(1,1))
        x_out = self.relu(x_merge)
        return x_out



class DropCatFusion(nn.Module):
    def __init__(self, in_channel_list, out_channel, drop_prob):
        super(DropCatFusion, self).__init__()
        self.in_channel_list = in_channel_list
        self.out_channel = out_channel
        self.drop_prob = drop_prob

        assert len(self.in_channel_list) >= 2

        s = 0
        for in_channel in self.in_channel_list:
            s = s + in_channel
        
        c_mid = max(s // 2, out_channel)
        self.merge_layer = nn.Sequential(
            nn.Dropout(p=self.drop_prob, inplace=False),
            backbone.conv1x1_bn_relu(s, c_mid),
            backbone.conv1x1_bn_relu(c_mid, out_channel)
        )
    
    def forward(self, x_list):
        #pdb.set_trace()
        x_merge = torch.cat(x_list, dim=1)
        x_out = self.merge_layer(x_merge)
        return x_out


class DropBranch(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropBranch, self).__init__()
        self.drop_prob = drop_prob
        assert (self.drop_prob >= 0) and (self.drop_prob <= 1)
        self.keep_prob = 1 - self.drop_prob
    
    def forward(self, x):
        if (self.drop_prob == 0) or (not self.training):
            return x
        else:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # per sample
            random_tensor = self.keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            output = x.div(self.keep_prob) * random_tensor
            return output


class DropBranchCatFusion(nn.Module):
    def __init__(self, in_channel_list, out_channel, drop_prob):
        super(DropBranchCatFusion, self).__init__()
        self.in_channel_list = in_channel_list
        self.out_channel = out_channel
        self.drop_prob = drop_prob

        assert len(self.in_channel_list) >= 2

        s = 0
        for in_channel in self.in_channel_list:
            s = s + in_channel
        
        c_mid = max(s // 2, out_channel)
        self.merge_layer = nn.Sequential(
            backbone.conv1x1_bn_relu(s, c_mid),
            backbone.conv1x1_bn_relu(c_mid, out_channel)
        )

        self.drop_branch_list = [DropBranch(self.drop_prob) for i in range(len(self.in_channel_list))]
    
    def forward(self, x_list):
        #pdb.set_trace()
        x_list_drop_branch = [self.drop_branch_list[i](x_list[i]) for i in range(len(self.in_channel_list))]
        x_merge = torch.cat(x_list_drop_branch, dim=1)
        x_out = self.merge_layer(x_merge)
        return x_out

class SpatialFusion(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(SpatialFusion, self).__init__()
        cin = sum(in_channel_list)
        self.sensor_gate = nn.Sequential(
                backbone.conv3x3(cin, cin, stride=1, dilation=1, bias=True),
                nn.Sigmoid()
            )
        self.point_layer = backbone.conv1x1_bn_relu(cin, out_channel)

    def forward(self, x):
        xi = torch.cat(x, 1)
        gate = self.sensor_gate(xi)
        xo = gate*xi
        xo = self.point_layer(xo)
        return xo

class PointFusion(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(PointFusion, self).__init__()
        cin = sum(in_channel_list)
        self.sensor_gate = nn.Sequential(
                backbone.conv1x1(cin, cin, bias=True),
                nn.Sigmoid()
            )
        self.point_layer = backbone.conv1x1_bn_relu(cin, out_channel)

    def forward(self, x):
        xi = torch.cat(x, 1)
        gate = self.sensor_gate(xi)
        xo = gate*xi
        xo = self.point_layer(xo)
        return xo

class AdaptionFusion(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channel_list,
                 out_channel,
                 kernel_size=3,
                 deformable_groups=4):
        super(AdaptionFusion, self).__init__()
        in_channels = sum(in_channel_list)
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            in_channels, deformable_groups * offset_channels, 1, bias=True)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channel,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()

    def forward(self, x,):
        x = torch.cat(x, 1)
        offset = self.conv_offset(x)
        x = self.relu(self.conv_adaption(x, offset))
        return x
    
class SpatialCatFusion(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(SpatialCatFusion, self).__init__()
        self.scale_factor = scale_factor
        self.cout = cout

        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.conv_layer = nn.Sequential(
            backbone.conv3x3_bn_relu(2 * cout, cout, stride=1, dilation=1),
        )

        self.conv_high = backbone.conv3x3_bn_relu(cin_high, cout, stride=1, dilation=1)
        self.conv_low = backbone.conv3x3_bn_relu(cin_low, cout, stride=1, dilation=1)
    
    def forward(self, x_list):
        #pdb.set_trace()
        x_low, x_high = x_list
        x_high_up = self.upsample(x_high)

        x_low_feat = self.conv_low(x_low)
        x_high_up_feat = self.conv_high(x_high_up)

        x_merge = torch.cat((x_low_feat, x_high_up_feat), dim=1) #(BS, 2*channels, H, W)
        x_merge = self.dropout(x_merge)
        x_out = self.conv_layer(x_merge)
        return x_out

class SpatialCatFusion1(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(SpatialCatFusion1, self).__init__()
        self.scale_factor = scale_factor
        self.cout = cout

        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.conv_layer = nn.Sequential(
            backbone.conv1x1_bn_relu(2 * cout, cout)
        )

        self.conv_high = backbone.conv3x3_bn_relu(cin_high, cout, stride=1, dilation=1)
        self.conv_low = backbone.conv3x3_bn_relu(cin_low, cout, stride=1, dilation=1)
    
    def forward(self, x_list):
        #pdb.set_trace()
        x_low, x_high = x_list
        x_high_up = self.upsample(x_high)

        x_low_feat = self.conv_low(x_low)
        x_high_up_feat = self.conv_high(x_high_up)

        x_merge = torch.cat((x_low_feat, x_high_up_feat), dim=1) #(BS, 2*channels, H, W)
        x_merge = self.dropout(x_merge)
        x_out = self.conv_layer(x_merge)
        return x_out

    
class SpatialCatFusion2(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(SpatialCatFusion2, self).__init__()
        self.scale_factor = scale_factor
        self.cout = cout

        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        self.conv_high = backbone.conv3x3_bn_relu(cin_high, cout//2, stride=1, dilation=1)
        self.conv_low = backbone.conv3x3_bn_relu(cin_low, cout//2, stride=1, dilation=1)
    
    def forward(self, x_list):
        #pdb.set_trace()
        x_low, x_high = x_list
        x_high_up = self.upsample(x_high)

        x_low_feat = self.conv_low(x_low)
        x_high_up_feat = self.conv_high(x_high_up)

        x_out = torch.cat((x_low_feat, x_high_up_feat), dim=1) #(BS, 2*channels, H, W)
        return x_out


class SpatialAttFusion(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(SpatialAttFusion, self).__init__()
        self.scale_factor = scale_factor
        self.cout = cout

        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.att_layer = nn.Sequential(
            backbone.conv3x3(cout*2, cout, stride=1, dilation=1, bias=True),
            nn.Sigmoid()
        )
        self.conv_high = backbone.conv3x3_bn_relu(cin_high, cout, stride=1, dilation=1)
        self.conv_low = backbone.conv3x3_bn_relu(cin_low, cout, stride=1, dilation=1)
    
    def forward(self, x_list):
        #pdb.set_trace()
        x_low, x_high = x_list
        x_high_up = self.upsample(x_high)

        x_low_feat = self.conv_low(x_low)
        x_high_up_feat = self.conv_high(x_high_up)

        x_merge = torch.cat((x_low_feat, x_high_up_feat), dim=1) #(BS, 2*channels, H, W)
        x_merge = self.dropout(x_merge)

        # attention fusion
        ca_map = self.att_layer(x_merge)
        x_out = x_low_feat * ca_map + x_high_up_feat * (1 - ca_map)
        return x_out
    
class SpatialSEFusion(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(SpatialSEFusion, self).__init__()
        self.scale_factor = scale_factor
        self.cout = cout

        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.att_layer = nn.Sequential(
            backbone.conv3x3(cout*2, cout*2, stride=1, dilation=1, bias=True),
            nn.Sigmoid()
        )
        self.conv_high = backbone.conv3x3_bn_relu(cin_high, cout, stride=1, dilation=1)
        self.conv_low = backbone.conv3x3_bn_relu(cin_low, cout, stride=1, dilation=1)
    
        self.point_layer = backbone.conv1x1_bn_relu(cout*2, cout)

    def forward(self, x_list):
        #pdb.set_trace()
        x_low, x_high = x_list
        x_high_up = self.upsample(x_high)

        x_low_feat = self.conv_low(x_low)
        x_high_up_feat = self.conv_high(x_high_up)

        x_merge = torch.cat((x_low_feat, x_high_up_feat), dim=1) #(BS, 2*channels, H, W)
        x_merge = self.dropout(x_merge)

        # attention fusion
        ca_map = self.att_layer(x_merge)
        x_merge = x_merge*ca_map
        x_out = self.point_layer(x_merge)
        return x_out

class SpatialDCNFusion(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(SpatialDCNFusion, self).__init__()
        kernel_size = 3
        deformable_groups = 1
        offset_channels = kernel_size * kernel_size * 2

        self.scale_factor = scale_factor
        self.cout = cout

        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.att_layer = nn.Sequential(
            backbone.conv3x3(cout*2, cout, stride=1, dilation=1, bias=True),
            nn.Sigmoid()
        )
        self.conv_high = backbone.conv3x3_bn_relu(cin_high, cout, stride=1, dilation=1)
        self.conv_low = backbone.conv3x3_bn_relu(cin_low, cout, stride=1, dilation=1)

        self.conv_offset = nn.Conv2d(cout*2, deformable_groups * offset_channels, 1, bias=True)
        self.conv_mask = nn.Conv2d(cout*2, kernel_size*kernel_size, 1, bias=True)
        self.conv = nn.Conv2d(cout*2, cout, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        nn.init.constant_(self.conv_mask.weight, 0.5)

    def forward(self, x_list):
        #pdb.set_trace()
        x_low, x_high = x_list
        x_high_up = self.upsample(x_high)

        x_low_feat = self.conv_low(x_low)
        x_high_up_feat = self.conv_high(x_high_up)

        x_merge = torch.cat((x_low_feat, x_high_up_feat), dim=1) #(BS, 2*channels, H, W)
        # x_merge = self.dropout(x_merge)

        offset = self.conv_offset(x_merge)
        mask = torch.sigmoid(self.conv_mask(x_merge))
        x_merge = deform_conv2d(x_merge, offset, weight=self.conv.weight, mask=mask, padding=(1,1))
        # print(x_merge.dtype, x_merge.shape, offset.dtype, offset.shape)
        x_out = self.relu(x_merge)
        return x_out
