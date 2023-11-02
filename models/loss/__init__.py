import torch
import torch.nn as nn
import torch.nn.functional as F

from . import lovasz_losses
from .centernet_loss import CenterDetLoss, _transpose_and_gather_feat
from .kd_loss import KDLogitLoss, KDHintLoss, KDLoss, NKDLoss


import numpy as np
import pdb


def get_ohem_loss(loss_mat, valid_mask=None, top_ratio=0, top_weight=1):
    loss_mat_valid = None
    valid_num = None
    topk_num = None
    if valid_mask is not None:
        loss_mat_valid = (loss_mat * valid_mask.float()).view(-1)
        valid_num = int(valid_mask.float().sum())
        topk_num = int(valid_num * top_ratio)
    else:
        loss_mat_valid = loss_mat.view(-1)
        valid_num = loss_mat_valid.shape[0]
        topk_num = int(valid_num * top_ratio)
    
    loss_total = loss_mat_valid.sum() / (valid_num + 1e-12)
    if topk_num == 0:
        return loss_total
    else:
        loss_topk = torch.topk(loss_mat_valid, k=topk_num, dim=0, largest=True, sorted=False)[0]
        loss_total = loss_total + top_weight * loss_topk.mean()
        return loss_total


class CE_OHEM(nn.Module):
    def __init__(self, top_ratio=0.3, top_weight=1.0, ignore_index=-1, class_weight=None, weight=1, is_mask=False):
        super(CE_OHEM, self).__init__()
        self.top_ratio = top_ratio
        self.top_weight = top_weight
        self.ignore_index = ignore_index
        self.class_weight = class_weight
        self.weight = weight
        self.is_mask = is_mask

        if self.class_weight is not None:
            self.class_weight = torch.FloatTensor(self.class_weight)
        self.loss_func = nn.CrossEntropyLoss(reduce=False, ignore_index=self.ignore_index)
    
    def forward(self, params, name='loss'):
        #pdb.set_trace()
        pred = params['pred']
        gt = params['target']

        if 'mask' in params:
            valid_mask = params['mask']
        elif self.is_mask:
            valid_mask = gt!=self.ignore_index
        else:
            valid_mask = None

        loss_mat = self.loss_func(pred, gt.long())
        loss_result = get_ohem_loss(loss_mat, valid_mask, top_ratio=self.top_ratio, top_weight=self.top_weight)
        loss_result = loss_result*self.weight
        return loss_result


class BCE_OHEM(nn.Module):
    def __init__(self, top_ratio=0.3, top_weight=1.0, weight=1):
        super(BCE_OHEM, self).__init__()
        self.top_ratio = top_ratio
        self.top_weight = top_weight
        self.weight = weight
    
    def forward(self, pred, gt, valid_mask=None):
        #pdb.set_trace()
        loss_mat = F.binary_cross_entropy(pred, gt, reduce=False)
        loss_result = get_ohem_loss(loss_mat, valid_mask, top_ratio=self.top_ratio, top_weight=self.top_weight)
        return loss_result * self.weight

class LovaszSoftmax(nn.Module):
    def __init__(self, ignore_index=-1, weight=1):
        super(LovaszSoftmax, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
    
    def forward(self, params, name='loss'):
        #pdb.set_trace()
        pred = params['pred']
        gt = params['target']
        loss = lovasz_losses.lovasz_softmax(pred, gt, ignore=self.ignore_index)
        loss = loss*self.weight
        return loss
    
class ConsistencyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(ConsistencyLoss, self).__init__()
        self.weight = weight

    def forward(self, params, name='loss'):
        pred = params['pred']
        pred_raw = params['pred_raw']

        pred_cls_softmax = F.softmax(pred, dim=1)
        pred_cls_raw_softmax = F.softmax(pred_raw.detach(), dim=1)

        loss = (pred_cls_softmax - pred_cls_raw_softmax).abs().sum(dim=1).mean()
        loss = loss*self.weight
        return loss
