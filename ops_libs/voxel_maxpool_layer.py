import torch
from torch import nn
from torch.autograd import Function

import point_ops.pytorch_plugins as pytorch_plugins

import numpy as np
import os

import pdb


__all__ = ['VoxelMaxPool']

# pcds_feat, (BS, C, N, 1)
# pcds_ind,(BS, N, D, 1), D -> d1, d2, ..., dn
# voxel_out, (BS, C, D1, D2, ..., Dn)
# voxel_max_idx, (BS, N)
class VoxelMaxPoolFunction(Function):
    @staticmethod
    def forward(ctx, pcds_feat, pcds_ind, output_size, scale_rate):
        assert(pcds_ind.dtype == torch.float)
        assert(pcds_feat.dim() == 4)
        assert(pcds_ind.dim() == 4)
        assert(pcds_feat.size(2) == pcds_ind.size(1))
        assert(pcds_ind.size(2) == len(output_size))
        assert(pcds_ind.size(2) == len(scale_rate))

        voxel_out_shape = [pcds_feat.size(0), pcds_feat.size(1)] + list(output_size)
        voxel_out = torch.zeros(voxel_out_shape, dtype=pcds_feat.dtype, device=pcds_feat.device)
        voxel_max_idx = torch.zeros([pcds_ind.size(0), pcds_ind.size(1)], dtype=torch.int32, device=pcds_feat.device)

        voxel_out_size_pt = torch.LongTensor(voxel_out_shape).to(device=pcds_feat.device, dtype=torch.int32)
        voxel_out_stride_pt = torch.LongTensor(voxel_out.stride()).to(device=pcds_feat.device, dtype=torch.int32)
        output_size_pt = voxel_out_size_pt[2:]
        scale_rate_pt = torch.FloatTensor(scale_rate).to(device=pcds_feat.device)
        
        ctx.use_cuda = pcds_feat.is_cuda
        if ctx.use_cuda:
            pytorch_plugins.voxel_maxpooling_forward(pcds_feat, pcds_ind, voxel_out, voxel_max_idx,
            voxel_out_size_pt, voxel_out_stride_pt, output_size_pt, scale_rate_pt)
        else:
            raise NotImplementedError
        
        ctx.input_shape = pcds_feat.shape
        ctx.save_for_backward(pcds_feat, pcds_ind, voxel_out, voxel_max_idx, voxel_out_size_pt, voxel_out_stride_pt, output_size_pt, scale_rate_pt)
        return voxel_out
    
    @staticmethod
    def backward(ctx, grad_voxel_out):
        pcds_feat, pcds_ind, voxel_out, voxel_max_idx, voxel_out_size_pt, voxel_out_stride_pt, output_size_pt, scale_rate_pt = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_voxel_out = grad_voxel_out.contiguous()
            grad_pcds_feat = torch.zeros(ctx.input_shape, dtype=grad_voxel_out.dtype, device=grad_voxel_out.device)
            if ctx.use_cuda:
                pytorch_plugins.voxel_maxpooling_backward(pcds_feat, pcds_ind, voxel_out, voxel_max_idx,
                grad_pcds_feat, grad_voxel_out, voxel_out_size_pt, voxel_out_stride_pt, output_size_pt, scale_rate_pt)
            else:
                raise NotImplementedError
            
            return grad_pcds_feat, None, None, None
        else:
            return None, None, None, None


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    return VoxelMaxPoolFunction.apply(pcds_feat, pcds_ind, output_size, scale_rate)