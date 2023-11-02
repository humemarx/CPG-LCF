import torch
from torch import nn
from torch.autograd import Function

import point_ops.pytorch_plugins as pytorch_plugins

import numpy as np
import os

import pdb


__all__ = ['Grid2Point']

# forward
# grid_in, (BS, C, H, W)
# pcds_ind,(BS, N, 2, 1), 2 -> h, w
# pcds_feat, (BS, C, N, 1)
class Grid2PointFunction(Function):
    @staticmethod
    def forward(ctx, grid_in, pcds_ind, scale_rate):
        assert(pcds_ind.dtype == torch.float)
        assert(grid_in.dim() == 4)
        assert(pcds_ind.dim() == 4)

        assert(pcds_ind.size(2) == 2)
        assert(len(scale_rate) == 2)

        pcds_feat = torch.zeros([grid_in.size(0), grid_in.size(1), pcds_ind.size(1), 1], dtype=grid_in.dtype, device=grid_in.device)
        
        grid_in_size_pt = torch.LongTensor(list(grid_in.shape)).to(device=grid_in.device, dtype=torch.int32)
        grid_in_stride_pt = torch.LongTensor(list(grid_in.stride())).to(device=grid_in.device, dtype=torch.int32)
        scale_rate_pt = torch.FloatTensor(scale_rate).to(device=grid_in.device)

        ctx.use_cuda = grid_in.is_cuda
        if ctx.use_cuda:
            pytorch_plugins.grid2point_forward(pcds_feat, pcds_ind, grid_in,
            grid_in_size_pt, grid_in_stride_pt, scale_rate_pt)
        else:
            raise NotImplementedError
        
        ctx.input_shape = grid_in.shape
        ctx.save_for_backward(pcds_ind, grid_in_size_pt, grid_in_stride_pt, scale_rate_pt)
        return pcds_feat
    
    @staticmethod
    def backward(ctx, grad_pcds_feat):
        pcds_ind, grid_in_size_pt, grid_in_stride_pt, scale_rate_pt = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_pcds_feat = grad_pcds_feat.contiguous()
            grad_grid_in = torch.zeros(ctx.input_shape, dtype=grad_pcds_feat.dtype, device=grad_pcds_feat.device)
            if ctx.use_cuda:
                pytorch_plugins.grid2point_backward(pcds_ind, grad_pcds_feat, grad_grid_in,
                grid_in_size_pt, grid_in_stride_pt, scale_rate_pt)
            else:
                raise NotImplementedError
            
            return grad_grid_in, None, None
        else:
            return None, None, None


def Grid2Point(grid_in, pcds_ind, scale_rate):
    return Grid2PointFunction.apply(grid_in, pcds_ind, scale_rate)