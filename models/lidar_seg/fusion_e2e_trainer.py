import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import collections
import numpy as np

from .fusion_seg_trainer import FusionSegTrainer
from .. import base_trainer
from ..base_trainer import gather_metric, all_gather
from ..model_utils import metric

from tqdm import tqdm
import shutil
import models.loss as loss_lib
import pdb

color_map = [[0, 0, 0],
     [245, 150, 100],
     [245, 230, 100],
     [150, 60, 30],
     [180, 30, 80],
     [255, 0, 0],
     [30, 30, 255],
     [200, 40, 255],
     [90, 30, 150],
     [255, 0, 255],
     [255, 150, 255],
     [75, 0, 75],
     [75, 0, 175],
     [0, 200, 255],
     [50, 120, 255],
     [0, 175, 0],
     [0, 60, 135],
     [80, 240, 150],
     [150, 240, 255],
     [0, 0, 255],
     [183, 130, 88],
     [220, 20, 60],
     [79, 210, 114], 
     [178, 90, 62]]


@torch.no_grad()
def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size

class FusionE2ETrainer(FusionSegTrainer): 

    def training_step(self, batch_idx, batch):
        '''
        Input:
            pcds_xyzi, pcds_xyzi_raw (BS, 7, N, 1), 7 -> (x, y, z, intensity, dist, diff_x, diff_y)
            pcds_coord, pcds_coord_raw (BS, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord, pcds_sphere_coord_raw (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
            pcds_target (BS, N, 1)
        '''
        pcds_xyzi = batch['pcds_xyzi']
        pcds_coord = batch['pcds_coord']
        pcds_sphere_coord = batch['pcds_sphere_coord']
        pcds_target = batch['pcds_target']

        if self.pGen.use_consistency:
            pcds_xyzi_raw = batch['pcds_xyzi_raw']
            pcds_coord_raw = batch['pcds_coord_raw']
            pcds_sphere_coord_raw = batch['pcds_sphere_coord_raw']

        batch_size = pcds_xyzi.shape[0]
        loss_sum = 0
        record_dic = collections.OrderedDict()
        if self.pModel.use_camera_raw:
            images = batch['images']
            point_coord = batch['point_coord']
            point_mask = batch['point_mask']

            pred_result = self.model(pcds_xyzi, 
                                     pcds_coord, 
                                     pcds_sphere_coord,
                                     images, 
                                     point_coord, 
                                     point_mask)

            if self.pGen.use_consistency:
                if 'point_coord_raw' in batch:
                    point_coord_raw = batch['point_coord_raw']
                    point_mask_raw = batch['point_mask_raw']
                else:
                    point_coord_raw = point_coord
                    point_mask_raw = point_mask
                
                if 'image_feature' in pred_result:
                    image_feature = pred_result['image_feature']
                else:
                    image_feature = None

                pred_result_raw = self.model(pcds_xyzi_raw, 
                                             pcds_coord_raw, 
                                             pcds_sphere_coord_raw,
                                             images, 
                                             point_coord_raw, 
                                             point_mask_raw,
                                             image_feature)
                pred_cls_raw_list = pred_result_raw['pred_cls_list']

            if 'LossCamera' in self.loss_funcs:
                point_img_preds = pred_result['point_img_preds']
                point_all_mask = torch.sum(point_mask, dim=1, keepdim=False).bool()
                point_all_mask = point_all_mask.unsqueeze(-1).contiguous()
                loss_param_img_dic = {
                    'pred': point_img_preds,
                    'target': pcds_target.cuda(),
                    'mask': point_all_mask.cuda(),
                }

                img_loss_func = self.loss_funcs['LossCamera']
                loss_img = img_loss_func(loss_param_img_dic, 'LossCamera')
                loss_sum += loss_img
                record_dic.update({'LossCamera': loss_img})

        else:
            pred_result = self.model(pcds_xyzi, 
                                     pcds_coord, 
                                     pcds_sphere_coord)
            
            if self.pGen.use_consistency:
                pred_result_raw = self.model(pcds_xyzi_raw, 
                                             pcds_coord_raw, 
                                             pcds_sphere_coord_raw)
                pred_cls_raw_list = pred_result_raw['pred_cls_list']

        pred_cls_list = pred_result['pred_cls_list']
        # each stage loss function
        for n in range(len(pred_cls_list)):
            loss_param_dic = {
                'pred': pred_cls_list[n],
                'target': pcds_target.cuda(),
            }

            if self.pGen.use_consistency:
                loss_param_dic2 = {
                    'target': pcds_target.cuda(),
                    'pred': pred_cls_raw_list[n],
                }

                loss_param_dic3 = {
                    'target': pcds_target.cuda(),
                    'pred': pred_cls_list[n],
                    'pred_raw': pred_cls_raw_list[n],
                }

            loss_stage = 0
            for loss_name, loss_func in self.loss_funcs.items():
                if loss_name == 'LossConsist' :
                    if self.pGen.use_consistency:
                        loss_consist = loss_func(loss_param_dic3, 'stage{}'.format(n))
                        record_dic.update({"stage{}/{}".format(n, loss_name): loss_consist.item()})
                        loss_stage += loss_consist
                elif loss_name == 'LossDet':
                    pass
                elif loss_name == 'LossCamera':
                    pass

                else:
                    loss_tmp = loss_func(loss_param_dic, 'stage{}'.format(n))
                    try:
                        record_dic.update({"stage{}/{}".format(n, loss_name): loss_tmp.item()})
                    except:
                        print(loss_param_dic['pred'],loss_param_dic['target'],loss_tmp)
                    loss_stage += 0.5*loss_tmp
                    if self.pGen.use_consistency:
                        loss_raw_tmp = loss_func(loss_param_dic2, 'stage{}'.format(n))
                        record_dic.update({"stage{}/{}_raw".format(n, loss_name): loss_raw_tmp.item()})
                        loss_stage += 0.5*loss_raw_tmp


            loss_sum += loss_stage

        # log
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        reduced_loss = reduce_tensor(loss_sum)
        if (batch_idx % self.pGen.log_frequency == 0) and self.global_rank == 0:
            string = 'Epoch: [{}]/[{}]; Iteration: [{}]/[{}]; lr: {}'.format(self.epoch_idx, self.max_epochs,\
                    batch_idx, self.train_iters, lr)
            
        for key, value in record_dic.items():
            if (batch_idx % self.pGen.log_frequency == 0) and self.global_rank == 0:
                self.writer.add_scalar(key, value, self.global_step_idx)
                string = string + '\n {}: {}'.format(key, value)

        if (batch_idx % self.pGen.log_frequency == 0) and self.global_rank == 0:
            string = string + '\n loss: {0}'.format(reduced_loss.item())
            self.writer.add_scalar('total_loss', reduced_loss.item(), self.global_step_idx)
            self.writer.add_scalar("learning_rate", lr, self.global_step_idx)
            self.logger.info(string)
        return loss_sum
    
    @torch.no_grad()
    def validation_step(self, batch_idx, batch):
        pcds_xyzi = batch['pcds_xyzi']
        pcds_coord = batch['pcds_coord']
        pcds_sphere_coord = batch['pcds_sphere_coord']
        pcds_target = batch['pcds_target']
        sample_token = batch['sample_token']
        lidar_token = batch['lidar_token']
        assert pcds_xyzi.shape[0] == 1
        pcds_xyzi = pcds_xyzi.squeeze(0)
        pcds_coord = pcds_coord.squeeze(0)
        pcds_sphere_coord = pcds_sphere_coord.squeeze(0)
        pcds_target = pcds_target.squeeze(0)[0].squeeze()

        if self.pModel.use_camera_raw:
            images = batch['images']
            point_coord = batch['point_coord']
            point_mask = batch['point_mask']

            pred_result = self.model(pcds_xyzi.cuda(), 
                                     pcds_coord.cuda(), 
                                     pcds_sphere_coord.cuda(),
                                     images.cuda(), 
                                     point_coord.cuda(), 
                                     point_mask.cuda())

        else:
            pred_result = self.model(pcds_xyzi.cuda(), 
                                     pcds_coord.cuda(), 
                                     pcds_sphere_coord.cuda())
        # forward
        pred_cls_list = pred_result['pred_cls_list']
        pred_cls_list = torch.cat(pred_cls_list, dim=-1) #(BS, C, N, K)
        pred_cls_list = F.softmax(pred_cls_list, dim=1)
        pred_cls_list = pred_cls_list.mean(dim=0).permute(2, 1, 0).contiguous() #(K, N, C)

        batch_result = {
            'pred_cls_map': pred_cls_list,
            'pcds_target': pcds_target
        }

        # add metric
        for metric_name, metric_func in self.metric_dics.items():
            metric_func.addBatch(batch_result)

    def validation_epoch_end(self):
        # metric statics
        epoch_metric_dics = {}
        for metric_name, metric_func in self.metric_dics.items():
            metric_func = gather_metric(metric_func)
            metric_cate = metric_func.get_metric()
            self.metric_dics[metric_name].reset()
            epoch_metric_dics.update({metric_name: metric_cate})

        if self.global_rank == 0:
            metric_value = epoch_metric_dics[self.monitor_keys['major']][self.monitor_keys['minor']]
            self.log_metric(epoch_metric_dics, self.epoch_idx)
            path_ckpt = os.path.join(self.model_prefix, '{}-{:.3f}-model.ckpt'.format(self.epoch_idx, metric_value))
            self.save_checkpoint(path_ckpt)

            if self.best_value <= metric_value:
                self.best_ckpt = path_ckpt
                self.best_value = metric_value
                self.best_epoch = self.epoch_idx
    
    @torch.no_grad()
    def test_step(self, batch_idx, batch):
        pcds_xyzi = batch['pcds_xyzi']
        pcds_coord = batch['pcds_coord']
        pcds_sphere_coord = batch['pcds_sphere_coord']
        sample_token = batch['sample_token']
        lidar_token = batch['lidar_token']
        assert pcds_xyzi.shape[0] == 1
        pcds_xyzi = pcds_xyzi.squeeze(0)
        pcds_coord = pcds_coord.squeeze(0)
        pcds_sphere_coord = pcds_sphere_coord.squeeze(0)

        if self.pModel.use_camera_raw:
            images = batch['images']
            point_coord = batch['point_coord']
            point_mask = batch['point_mask']

            pred_result = self.model(pcds_xyzi.cuda(), 
                                     pcds_coord.cuda(), 
                                     pcds_sphere_coord.cuda(),
                                     images.cuda(), 
                                     point_coord.cuda(), 
                                     point_mask.cuda())

        else:
            pred_result = self.model(pcds_xyzi.cuda(), 
                                     pcds_coord.cuda(), 
                                     pcds_sphere_coord.cuda())

        # forward
        pred_cls = pred_result['pred_cls_list'][-1]#(BS, C, N)
        pred_cls = pred_cls.squeeze(-1)
        pred_cls = F.softmax(pred_cls, dim=1)
        pred_cls = pred_cls.mean(dim=0).permute(1, 0).contiguous() #(N, C)
        _, pred_map = torch.max(pred_cls, dim=1)
        pred_map_np = pred_map.cpu().numpy().astype(np.uint8) #(K, N)
        test_save_path = os.path.join(self.save_path, 'lidarseg', 'test')

        if not os.path.exists(test_save_path):
            os.system("mkdir -p {}".format(test_save_path))

        result_path = os.path.join(test_save_path, '{}_lidarseg.bin'.format(lidar_token[0]))
        pred_map_np.tofile(result_path)
        
    def test_epoch_end(self):
        pass