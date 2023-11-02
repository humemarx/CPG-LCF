import numpy as np
import torch
import collections
import pdb


class MultiClassMetric:
    def __init__(self, ignore_index=-1, Classes=[], stage=0, metric_key='mIOU'):
        self.ignore_index = ignore_index
        self.Classes = Classes
        self.stage = stage
        self.metric_key = metric_key
        self.all_tensors = {}
        self.reset()
    
    def reset(self):
        self.tp = np.zeros(len(self.Classes), dtype=np.float32)
        self.pred_num = np.zeros(len(self.Classes), dtype=np.float32)
        self.gt_num = np.zeros(len(self.Classes), dtype=np.float32)
        self.all_tensors = {}
    
    def addBatch(self, batch_result):
        # gt, pred (N)
        gt_map = batch_result['pcds_target'].cpu()
        pred_map = batch_result['pred_cls_map'][self.stage].cpu()

        _, pred_map = torch.max(pred_map, dim=-1)
        gt_map = (gt_map.float()).data.cpu().numpy()
        pred_map = (pred_map.float()).data.cpu().numpy()

        ignore_mask = (gt_map == self.ignore_index)
        
        gt_map[ignore_mask] = -1
        pred_map[ignore_mask] = -1
        for i, cate in enumerate(self.Classes):
            if i != self.ignore_index:
                pred_tmp = (pred_map == i).astype(np.float32)
                gt_tmp = (gt_map == i).astype(np.float32)
                
                #pdb.set_trace()
                tp = (pred_tmp * gt_tmp).sum()
                pred_num = pred_tmp.sum()
                gt_num = gt_tmp.sum()
                
                self.tp[i] = self.tp[i] + tp
                self.pred_num[i] = self.pred_num[i] + pred_num
                self.gt_num[i] = self.gt_num[i] + gt_num
    
        self.all_tensors.update({'tp': self.tp})
        self.all_tensors.update({'pred_num': self.pred_num})
        self.all_tensors.update({'gt_num': self.gt_num})

    def get_metric(self):
        result_dic = collections.OrderedDict()
        self.tp = self.all_tensors['tp']
        self.pred_num = self.all_tensors['pred_num']
        self.gt_num = self.all_tensors['gt_num']

        iou = self.tp / (self.gt_num + self.pred_num - self.tp + 1e-12)
        pre = self.tp / (self.pred_num + 1e-12)
        rec = self.tp / (self.gt_num + 1e-12)
        
        iou_mean = []
        for i, cate in enumerate(self.Classes):
            if i != self.ignore_index:
                result_dic[cate + '_iou'] = iou[i]
                result_dic[cate + '_pre'] = pre[i]
                result_dic[cate + '_rec'] = rec[i]
                iou_mean.append(iou[i])
        valid_mask = (self.gt_num > 0).astype(np.float32)
        result_dic[self.metric_key] = (iou * valid_mask).sum() / (valid_mask.sum() + 1e-12)
        self.reset()
        return result_dic
