# coding=utf-8
'''
Author: husserl
License: Apache Licence
Software: VSCode
Date: 2023-06-15 08:00:10
LastEditors: husserl
LastEditTime: 2023-06-21 11:28:59
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from models.utils.fp16_utils import force_fp32

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

class RegLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''
    def __init__(self, weight=1.0, code_weights=[]):
        super(RegLoss, self).__init__()
        self.weight = weight
        self.code_weights = code_weights
  
    def forward(self, output, mask, ind, target):
        if target.numel() == 0:
            return output.sum() * 0

        pred = _transpose_and_gather_feat(output, ind)
        # pred: batch x max_objects x dim
        mask = mask.float().unsqueeze(2) 
        obj_num = (mask.sum() + 1e-4)
        loss = F.l1_loss(pred*mask, target*mask, reduction='none')
        loss = loss.transpose(2, 0).sum(dim=2).sum(dim=1)
        # loss = (loss*loss.new_tensor(self.code_weights)).sum()*self.weight/obj_num
        loss = loss*loss.new_tensor(self.code_weights)*self.weight/obj_num
        return loss

class FastFocalLoss(nn.Module):
    '''
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    '''
    def __init__(self, weight=1.0, alpha=2.0, gamma=4.0):
        super(FastFocalLoss, self).__init__()
        self.weight = weight
        self.alpha=alpha
        self.gamma=gamma

    def forward(self, pred, target, ind, mask, cat):
        '''
        Arguments:
          out, target: B x C x H x W
          ind, mask: B x M
          cat (category id for peaks): B x M
        '''
        eps = 1e-12
        mask = mask.float()
        gt = torch.pow(1 - target, self.gamma)
        neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, self.alpha) * gt
        neg_loss = neg_loss.sum()

        pos_pred_pix = _transpose_and_gather_feat(pred, ind) # B x M x C
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
        num_pos = mask.sum()
        pos_loss = torch.log(pos_pred + eps) * torch.pow(1 - pos_pred, self.alpha) * mask.unsqueeze(2)
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return -neg_loss*self.weight
        return -(pos_loss + neg_loss) *self.weight / num_pos


class CenterDetLoss(nn.Module):
    def __init__(self, det_weight=1.0, loc_weight=0.25, hm_weight=1.0, code_weights=[]):
        super(CenterDetLoss, self).__init__()
        self.det_weight = det_weight
        self.loc_weight = loc_weight
        self.hm_weight = hm_weight
        self.code_weights = code_weights
        self.crit = FastFocalLoss(weight=hm_weight, alpha=2.0, gamma=4.0)
        self.crit_reg = RegLoss(weight=loc_weight, code_weights=code_weights)

    @force_fp32(apply_to=('preds_dicts', 'gt_dicts'))
    def forward(self, preds_dicts, gt_dicts, name='Det'):
        total_loss = 0.0
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = torch.clamp(preds_dict['hm'].sigmoid_(), min=1e-4, max=1-1e-4)
            # preds_dict['hm'] = torch.clamp(preds_dict['hm'].sigmoid_(), min=1e-4, max=1-1e-4)
            num_pos = gt_dicts['mask'][task_id].float().sum().item()

            hm_loss = self.crit(preds_dict['hm'], gt_dicts['hm'][task_id], gt_dicts['ind'][task_id], gt_dicts['mask'][task_id], gt_dicts['cat'][task_id])

            target_box = gt_dicts['anno_box'][task_id]
            # reconstruct the anno_box from multiple reg heads
            if 'vel' in preds_dict:
                preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                    preds_dict['vel'], preds_dict['rot']), dim=1)  
            else:
                preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                    preds_dict['rot']), dim=1)   
                target_box = target_box[..., [0, 1, 2, 3, 4, 5, -2, -1]] # remove vel target                       

            # Regression loss for dimension, offset, height, rotation            
            box_loss = self.crit_reg(preds_dict['anno_box'], gt_dicts['mask'][task_id], gt_dicts['ind'][task_id], target_box)
            loc_loss = box_loss.sum()

            total_loss += (hm_loss + loc_loss)*self.det_weight
            loss_dict.update({'{}/hm_loss/{}'.format(name, task_id): hm_loss.item(), 
                              '{}/loc_loss/{}'.format(name, task_id): loc_loss.item(), 
                              '{}/reg_loss/{}'.format(name, task_id): box_loss[0:3].mean().item(),
                              '{}/dim_loss/{}'.format(name, task_id): box_loss[3:6].mean().item()})

            if 'vel' in preds_dict:
                loss_dict.update({'{}/vel_loss/{}'.format(name, task_id): box_loss[6:8].mean().item(),
                                  '{}/rot_loss/{}'.format(name, task_id): box_loss[8:10].mean().item()})
            else:
                loss_dict.update({'{}/rot_loss/{}'.format(name, task_id): box_loss[6:8].mean().item()})

            loss_dict.update({'{}/num_pos/{}'.format(name, task_id): num_pos})
        
        return total_loss, loss_dict
