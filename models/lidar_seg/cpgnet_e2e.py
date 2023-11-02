import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import collections
import numpy as np

from ops_libs import Grid2Point, VoxelMaxPool, max_op

from .. import networks
from utils.config_parser import get_module

import pdb


class CPGNet(nn.Module):
    def __init__(self, pModel):
        super(CPGNet, self).__init__()
        self.pModel = pModel
        self.build_network()
    
    def build_network(self):
        self.bev_shape = list(self.pModel.Voxel.bev_shape)
        self.rv_shape = list(self.pModel.Voxel.rv_shape)
        self.bev_wl_shape = self.bev_shape[:2]

        self.point_feat_out_channels = self.pModel.point_feat_out_channels
        self.stage_num = len(self.point_feat_out_channels)

        # build cascaded network
        self.pre_modal_list = nn.ModuleList()
        self.lidar_net_list = nn.ModuleList()
        self.modal_net_list = nn.ModuleList()
        self.modal_fusion_list = nn.ModuleList()
        self.bev_net_list = nn.ModuleList()
        self.rv_net_list = nn.ModuleList()
        self.seghead_vf_list = nn.ModuleList()
        self.seghead_pred_list = nn.ModuleList()
        self.bev_seg_list = nn.ModuleList()
        self.rv_seg_list = nn.ModuleList()
        if hasattr(self.pModel, 'cam_fusion'):
            self.cam_fusion = self.pModel.cam_fusion
        else:
            self.cam_fusion = 'sum'

        stage_feat_num = 0
        if self.pModel.with_bev:
            bev_net_cfg = self.pModel.BEVParam
            stage_feat_num = bev_net_cfg.base_channels[0]

        if self.pModel.with_rv:
            rv_net_cfg = self.pModel.RVParam
            stage_feat_num = rv_net_cfg.base_channels[0]

        fusion_cfg = self.pModel.FusionParam

        # image branch
        if self.pModel.use_camera_raw:
            self.image_net = get_module(self.pModel.ImageBranch)
            self.image_pred = networks.backbone.PredBranch(self.pModel.ModalParam.modal_nums[-1], self.pModel.class_num)

        # stage
        point_feat_in_channel = self.pModel.ModalParam.lidar_num
        for n in range(0, self.stage_num):
            # define stage n network
            if n==0:
                self.lidar_net_list.append(networks.backbone.bn_conv1x1_bn_relu(point_feat_in_channel, stage_feat_num))
            else:
                self.lidar_net_list.append(networks.backbone.conv1x1_bn_relu(point_feat_in_channel, stage_feat_num))

            modal_nets = nn.ModuleList()
            modal_channels_n = [stage_feat_num]
            for idx, modal_name in enumerate(self.pModel.ModalParam.modal_list):
                modal_nets.append(
                    nn.Sequential(
                        networks.backbone.conv1x1_bn_relu(self.pModel.ModalParam.modal_nums[idx], stage_feat_num),
                        networks.backbone.conv1x1_bn_relu(stage_feat_num, stage_feat_num),
                    )        
                )
                modal_channels_n.append(stage_feat_num)
            self.modal_net_list.append(modal_nets)
            # multi-modal fusion
            self.modal_fusion_list.append(get_module(fusion_cfg, modal_channels_n, stage_feat_num))

            # multi-view fusion
            point_fusion_channels = [stage_feat_num]
            if self.pModel.with_bev:
                self.bev_net_list.append(get_module(bev_net_cfg))
                point_fusion_channels.append(self.bev_net_list[n].out_channels)

            if self.pModel.with_rv:
                self.rv_net_list.append(get_module(rv_net_cfg))
                point_fusion_channels.append(self.rv_net_list[n].out_channels)

            self.seghead_vf_list.append(get_module(fusion_cfg, in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels[n]))
            point_feat_in_channel = self.point_feat_out_channels[n]
            # seg prediction
            self.seghead_pred_list.append(networks.backbone.PredBranch(self.point_feat_out_channels[n], self.pModel.class_num))

    def image_process(self, images, point_coord, point_mask, use_image_feature=None):
        '''
        description: image raw data process
        Args:
            images: Batch_size, camera_nums, H, W, Channels
            point_coord: Batch_size, camera_nums, point_nums, 2  --> (x, y)
            point_mask: Batch_size, camera_nums, point_nums
        return {*}
        '''        
        BS, N, H, W, C = images.shape  # BS, number_camera, H, W, Channel
        if use_image_feature is None:
            images = images.permute(0, 1, 4, 2, 3).reshape(-1, C, H, W).contiguous()
            image_feature = self.image_net(images)
        else:
            image_feature = use_image_feature
        
        point_coord = point_coord[:, :, :, [1,0]] # ---> x,y to y,x
        _, feats_channels, h, w = image_feature.shape

        scale = (h / H, w / W)
        point_feature_camera = self.camera2point_grid2point_parral(image_feature, 
                                                                   point_coord, 
                                                                   point_mask,
                                                                   scale)
        return point_feature_camera, image_feature

    def camera2point_grid2point_parral(self, 
                                        image_feature, 
                                        point_coords, 
                                        point_masks, 
                                        scale):
        '''
        description: get camera feature of each point, use Grid2Point
        Args:
            image_feature: Batch_size x camera_nums, H, W, Channels
            point_coords: Batch_size, camera_nums, point_nums, 2
            point_masks: Batch_size, camera_nums, point_nums
            scale: 
        return:
            point_features: BS, point_nums, feature_channel
        '''        
        BS, C, Num_point, coord_num = point_coords.shape
        point_coords = point_coords.reshape(-1, Num_point, coord_num).unsqueeze(-1)
        upsample_features = Grid2Point(image_feature,
                                       point_coords,
                                       scale)  # bs, c, n, 1
        
        point_masks = point_masks.reshape(-1, 1, Num_point).unsqueeze(-1)
        point_masks = point_masks.repeat(1, upsample_features.shape[1], 1, 1)
        point_features = torch.zeros_like(upsample_features)
        point_features[point_masks] = upsample_features[point_masks]

        _, feature_dim, Num_point = point_features.squeeze(-1).shape
        point_features = point_features.squeeze(-1).reshape(BS, C, feature_dim, Num_point)
        if self.cam_fusion == 'sum':
            point_features = torch.sum(point_features, dim=1, keepdim=False)
            point_features = point_features.unsqueeze(-1).contiguous()
        elif self.cam_fusion == 'max':
            point_features = torch.max(point_features, dim=1, keepdim=False)[0]
            point_features = point_features.unsqueeze(-1).contiguous()
        elif self.cam_fusion == 'cat':
            point_features = point_features.reshape(BS, C*feature_dim, Num_point).contiguous()
        else:
            point_features = torch.sum(point_features, dim=1, keepdim=False)
            point_features = point_features.unsqueeze(-1).contiguous()
        return point_features

    def stage_n_forward(self, point_feat, point_modal_inputs, pcds_coord_wl, pcds_sphere_coord, stage_index=0):
        '''
        Input:
            point_feat (BS, C, N, 1)
            pcds_coord_wl (BS, N, 2, 1), 2 -> (x_quan, y_quan)
            pcds_sphere_coord (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
            stage_index, type: int, means the (stage_index).{th} stage forward
        Output:
            point_feat_out (BS, C1, N, 1)
        '''
        point_feat_lidar = self.lidar_net_list[stage_index](point_feat)

        point_feat_input = [point_feat_lidar]

        for idx, modal_feat in enumerate(point_modal_inputs):
            point_feat_modal = self.modal_net_list[stage_index][idx](modal_feat)
            point_feat_input.append(point_feat_modal)

        # fusion modals feats
        point_feat_fusion = self.modal_fusion_list[stage_index](point_feat_input)

        #merge multi-view
        if stage_index == 0:
            post_feat_inputs = [point_feat_lidar]
        else:
            post_feat_inputs = [point_feat]

        pred_out_list = []
        #bird-view
        bev_det_feats = []
        if self.pModel.with_bev:
            bev_input = VoxelMaxPool(pcds_feat=point_feat_fusion, pcds_ind=pcds_coord_wl, output_size=self.bev_wl_shape, scale_rate=(1.0, 1.0))
            bev_feat = self.bev_net_list[stage_index](bev_input)
            point_bev_feat = Grid2Point(bev_feat, pcds_coord_wl, scale_rate=self.pModel.BEVParam.scale_rate_list[stage_index])
            post_feat_inputs.append(point_bev_feat)

        #range-view
        rv_det_feats = []
        if self.pModel.with_rv:
            rv_input = VoxelMaxPool(pcds_feat=point_feat_fusion, pcds_ind=pcds_sphere_coord, output_size=self.rv_shape, scale_rate=(1.0, 1.0))
            rv_feat = self.rv_net_list[stage_index](rv_input)
            point_rv_feat = Grid2Point(rv_feat, pcds_sphere_coord, scale_rate=self.pModel.RVParam.scale_rate_list[stage_index])
            post_feat_inputs.append(point_rv_feat)

        #merge multi-view
        point_feat_out = self.seghead_vf_list[stage_index](post_feat_inputs)
        pred_out_list.append(point_feat_out)

        return pred_out_list

    def infer(self, pcds_xyzi, pcds_coord, pcds_sphere_coord):
        '''
        Input:
            pcds_xyzi (BS, 7, N, 1), 7 -> (x, y, z, intensity, dist, diff_x, diff_y)
            pcds_coord (BS, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
        Output:
            pred_cls
        '''
        pcds_coord_wl = pcds_coord[:, :, :2].contiguous()
        # split lidar and other modals
        lidar_feat = pcds_xyzi[:,:self.pModel.ModalParam.lidar_num]
        point_modal_inputs = []
        start_nums = self.pModel.ModalParam.lidar_num

        for idx, modal_name in enumerate(self.pModel.ModalParam.modal_list):
            end_nums = start_nums+self.pModel.ModalParam.modal_nums[idx]
            modal_feat = pcds_xyzi[:,start_nums:end_nums]
            point_modal_inputs.append(modal_feat)
            start_nums = end_nums

        # lidar_feat = self.pre_point_net(lidar_feat)
        # for idx, modal_feat in enumerate(point_modal_inputs):
        #     point_modal_inputs[idx] = self.pre_modal_list[idx](modal_feat)

        pcds_feat = lidar_feat
        stage_num = 1
        # each stage forward
        bev_feat_list = []
        rv_feat_list = []
        for n in range(stage_num):
            pcds_feat, bev_feat, rv_feat = self.stage_n_forward(pcds_feat, point_modal_inputs, pcds_coord_wl, pcds_sphere_coord, stage_index=n)
            bev_feat_list.extend(bev_feat)
            rv_feat_list.extend(rv_feat)

        pred_det_dicts = self.det_head(bev_feat_list)
        pred_cls = self.seghead_pred_list[stage_num-1](pcds_feat).squeeze(-1).permute(0, 2, 1).contiguous() #(BS, N, C)
        pred_cls = F.softmax(pred_cls, dim=-1) #(BS, N, C)
        max_score, max_index = max_op(pred_cls, dim=-1) #(BS, N)
        return pred_cls, max_score, max_index
    
    def forward(self, 
                pcds_xyzi, 
                pcds_coord, 
                pcds_sphere_coord,
                images=None, 
                point_coord=None, 
                point_mask=None,
                use_image_feature=None):
        '''
        Input:
            pcds_xyzi (BS, 7, N, 1), 7 -> (x, y, z, intensity, dist, diff_x, diff_y)
            pcds_coord (BS, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
        Output:
            pred_cls_list, list of pytorch tensors
        '''
        result = {}
        if self.pModel.use_camera_raw:
            point_feature_camera, image_feature = self.image_process(images, point_coord, point_mask, use_image_feature)
            point_img_preds = self.image_pred(point_feature_camera)
            if pcds_xyzi.shape[0] != point_feature_camera.shape[0]:
                assert point_feature_camera.shape[0] == 1
                point_feature_camera = point_feature_camera.repeat(pcds_xyzi.shape[0], 1, 1, 1)
                pcds_xyzi = torch.cat([pcds_xyzi, point_feature_camera], dim=1)
            else:
                pcds_xyzi = torch.cat([pcds_xyzi, point_feature_camera], dim=1)
            result.update({
                'point_img_preds': point_img_preds,
                'image_feature': image_feature
            })

        pcds_coord_wl = pcds_coord[:, :, :2].contiguous()

        lidar_feat = pcds_xyzi[:,:self.pModel.ModalParam.lidar_num]
        start_nums = self.pModel.ModalParam.lidar_num
        point_modal_inputs = []
        for idx, modal_name in enumerate(self.pModel.ModalParam.modal_list):
            end_nums = start_nums+self.pModel.ModalParam.modal_nums[idx]
            modal_feat = pcds_xyzi[:,start_nums:end_nums]
            point_modal_inputs.append(modal_feat)
            start_nums = end_nums

        pcds_feat_history = [lidar_feat]
        pred_cls_list = []
        bev_feat_list = []
        rv_feat_list = []
        # each stage forward
        for n in range(self.stage_num):
            stage_feat =  pcds_feat_history[n]

            pred_feat_list = self.stage_n_forward(stage_feat, point_modal_inputs, pcds_coord_wl, pcds_sphere_coord, stage_index=n)

            pcds_feat_history.append(pred_feat_list[-1])

            pred_cls_n = self.seghead_pred_list[n](pred_feat_list[-1]).float() #(BS, class_num, N, 1)
            pred_cls_list.append(pred_cls_n)

        result.update({
            'pred_cls_list': pred_cls_list,
        })
        return result