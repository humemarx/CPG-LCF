# coding=utf-8
'''
Author: husserl
License: Apache Licence
Software: VSCode
Date: 2023-08-18 09:21:35
LastEditors: husserl
LastEditTime: 2023-11-02 08:28:56
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLogitLoss(nn.Module):
    def __init__(self, temperature=10.0, weight=1.0):
        super(KDLogitLoss, self).__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(self, params, name='loss'):
        pred = params['pred']
        pred_raw = params['pred_raw'].detach()

        Si = F.log_softmax(pred/self.temperature, dim=1)
        Ti = F.softmax(pred_raw/self.temperature, dim=1)
        kd_loss = - (Ti*Si).sum(dim=-1).mean()
        return kd_loss*self.weight

class KDHintLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(KDHintLoss, self).__init__()
        self.weight = weight

    def forward(self, params, name='loss'):
        feat = params['feat']
        feat_raw = params['feat_raw']
        hint_loss = F.mse_loss(feat, feat_raw.detach(), reduction='mean')
        return hint_loss*self.weight

class KDLoss(nn.Module):
    def __init__(self, kd_t=1.0, kd_logit_w=1.0, kd_hint_w=1.0, weight=1.0):
        super(KDLoss, self).__init__()
        self.weight = weight
        self.kd_t = kd_t
        self.kd_logit_w = kd_logit_w
        self.kd_hint_w = kd_hint_w

    def forward(self, params, name='loss'):
        pred = params['pred']
        pred_raw = params['pred_raw'].detach()

        Si = F.log_softmax(pred/self.kd_t, dim=1)
        Ti = F.softmax(pred_raw/self.kd_t, dim=1)
        kd_logit_loss = - (Ti*Si).sum(dim=-1).mean()
        feat = params['feat']
        feat_raw = params['feat_raw'].detach()
        kd_hint_loss = F.mse_loss(feat, feat_raw, reduction='mean')
        kd_loss = kd_logit_loss*self.kd_logit_w + kd_hint_loss*self.kd_hint_w
        return kd_loss*self.weight

class NKDLoss(nn.Module):

    """ PyTorch version of NKD """

    def __init__(self, temp=1.0, gamma=1.5, weight=1.0):
        super(NKDLoss, self).__init__()

        self.temp = temp
        self.gamma = gamma
        self.weight = weight

    def forward(self, params, name='loss'):
        logit_s = params['pred']
        logit_t = params['pred_raw'].detach()
        gt_label = params['target']
        Bs, C, Pn, _ = logit_s.shape
        logit_s = logit_s.permute(0, 2, 1, 3).reshape(Bs*Pn, C).contiguous()
        logit_t = logit_t.permute(0, 2, 1, 3).reshape(Bs*Pn, C).contiguous()
        gt_label = gt_label.reshape(-1, 1).contiguous()

        if len(gt_label.size()) > 1:
            label = torch.max(gt_label, dim=1, keepdim=True)[1]
        else:
            label = gt_label.view(len(gt_label), 1)

        # N*class
        N, c = logit_s.shape
        s_i = F.log_softmax(logit_s, dim=1)
        t_i = F.softmax(logit_t, dim=1)
        # N*1
        s_t = torch.gather(s_i, 1, label)
        t_t = torch.gather(t_i, 1, label)

        loss_t = - (t_t * s_t).mean()

        mask = torch.ones_like(logit_s).scatter_(1, label, 1).bool()
        logit_s = logit_s[mask].reshape(N, -1)
        logit_t = logit_t[mask].reshape(N, -1)
        
        # N*class
        S_i = F.log_softmax(logit_s/self.temp, dim=1)
        T_i = F.softmax(logit_t/self.temp, dim=1)     

        loss_non =  (T_i * S_i).sum(dim=1).mean()
        loss_non = - self.gamma * loss_non

        return (loss_t + loss_non)*self.weight