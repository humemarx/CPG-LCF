# coding=utf-8
'''
Author: husserl
License: Apache Licence
Software: VSCode
Date: 2023-03-01 03:40:26
LastEditors: husserl
LastEditTime: 2023-11-02 09:18:48
'''
import pickle as pkl
from torch.utils.data import Dataset
import yaml
import json
from datasets import data_aug, utils, copy_paste, camera_aug
import numpy as np
import os
import os.path as osp
import copy
from nuscenes.utils.geometry_utils import view_points
import random
import math
import torch
from PIL import Image

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

class DataloadTrain(Dataset):
    def __init__(self,config):
        self.config = config
        self.mode = config.mode
        self.fname_pkl = config.fname_pkl
        self.data_root = config.SeqDir
        self.frame_point_num = random.choice(self.config.frame_point_num)
        with open('datasets/nusc/nuscenes.yaml', 'r') as f:
            self.task_cfg = yaml.load(f, Loader=yaml.Loader)

        # prob resample
        if hasattr(self.config, 'use_prob_resample'):
            self.use_prob_resample = self.config.use_prob_resample
        else:
            self.use_prob_resample = False

        self.use_camera = 'none'
        self.rand_level = 0

        self.point_aug = None
        self.image_aug = None

        self.init_lidar_aug()
        self.init_cp_aug()
        self.init_cam_anno()
        self.load_infos(self.fname_pkl)

    def init_cp_aug(self):
        print('init copy paste aug!')
        self.cp_aug = None
        if hasattr(self.config, 'CopyPasteAug') and self.config.CopyPasteAug.is_use:
            self.cp_aug = copy_paste.CutPaste(self.config.CopyPasteAug)

    def init_cam_anno(self):
        if hasattr(self.config, 'rand_level'):
            self.rand_level = self.config.rand_level

        print('init cam anno!')
        self.img_feat_num = 0
        # load image data
        if 'camera_raw' in self.config.SensorParam.modal_list:
            self.use_camera = 'camera_raw'
            self.img_feat_num = self.config.SensorParam.camera_feat_num

            transforms = []
            if hasattr(self.config, 'CameraAug'):
                for aug_dic in self.config.CameraAug.transforms:
                    aug_func = eval('camera_aug.{}'.format(aug_dic['type']))(**aug_dic['params'])
                    transforms.append(aug_func)
            self.image_aug = camera_aug.ImageAugCompose(transforms)
        else:
            pass

    def init_lidar_aug(self):
        print('init lidar aug!')
        if hasattr(self.config, 'PointAug'):
            transforms = []
            for aug_dic in self.config.PointAug.transforms:
                aug_func = eval('data_aug.{}'.format(aug_dic['type']))(**aug_dic['params'])
                transforms.append(aug_func)
            self.point_aug = data_aug.PointAugCompose(transforms)

    def load_infos(self, info_path):
        print('load data infos!')
        with open(info_path, 'rb') as f:
            self.data_infos = pkl.load(f)['infos']
        self.sample_length = len(self.data_infos)
        print('{} Samples: '.format(self.mode), self.sample_length)

        if hasattr(self.config, 'obj_sample') and self.config.obj_sample:
            # get object class dist
            _cls_infos = {name: [] for name in nus_categories}
            for info in self.data_infos:
                for name in set(info["gt_names"]):
                    if name in nus_categories:
                        _cls_infos[name].append(info)

            duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
            _cls_dist = {k: len(v) / max(duplicated_samples, 1) for k, v in _cls_infos.items()}

            self._nusc_infos_all = []
            frac = 1.0 / len(nus_categories)
            ratios = [frac / v for v in _cls_dist.values()]

            for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
                self._nusc_infos_all += np.random.choice(
                    cls_infos, int(len(cls_infos) * ratio)
                ).tolist()

            self.sample_length = len(self._nusc_infos_all)
            print('{} RE Samples: '.format(self.mode), self.sample_length)

        else:
            self._nusc_infos_all = self.data_infos

        # random.shuffle(self._nusc_infos_all)
        # self.data_infos = self._nusc_infos_all[:self.sample_length]
        self.data_infos = self._nusc_infos_all

    def load_pcd_from_file(self, file_path):
        file_path = os.path.join('data', file_path)
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)[:, :4]
        return points
    
    def load_pcdlabel_from_file(self, file_path):
        file_path = os.path.join('data', file_path)
        pcds_label_use = np.fromfile(file_path, dtype=np.uint8).reshape((-1))
        pcds_label_use = utils.relabel(pcds_label_use, self.task_cfg['learning_map'])
        return pcds_label_use

    def load_lidar_anno(self, info, with_bbox=False):
        lidarseg_path = info['lidarseg_path']
        pcds_label_use = self.load_pcdlabel_from_file(lidarseg_path)
        if with_bbox:
            gt_boxes = info['gt_boxes']  # x, y, z, l, w, h, yaw
            gt_names = info['gt_names']
            gt_velocity = info['gt_velocity']  # vx, vy
            num_lidar_pts = info['num_lidar_pts']
            num_radar_pts = info['num_radar_pts']
            valid_flag = info['valid_flag']
            name_mask = np.array([n in nus_categories for n in gt_names], dtype=np.bool_)
            valid_mask = valid_flag & name_mask

            gt_boxes = gt_boxes[valid_mask,:]
            gt_names = gt_names[valid_mask]
            gt_classes = np.array(
                [nus_categories.index(n) + 1 for n in gt_names],
                dtype=np.int32,
            )
            gt_velocity = gt_velocity[valid_mask,:]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_boxes = np.concatenate([gt_boxes, gt_classes[:,np.newaxis], gt_velocity[:, :2]], axis=1)
        else:
            gt_boxes = None
        # box: x,y,z,l,w,h,yaw,id,vx,vy
        return pcds_label_use, gt_boxes

    # load camera information
    def load_camera_image(self, cam_infos, pcds=None):
        camera_dict = {}
        for cam_type, cam_info in cam_infos.items():
            cam_path = cam_info['data_path']
            cam_path = os.path.join('data', cam_path)
            cam2lidar_mat = cam_info['sensor2lidar_mat']
            cam_intrinsic = cam_info['cam_intrinsic']
            cam_token = cam_info['sample_data_token']
            image = Image.open(cam_path)  # RGB, 
            size_w, size_h = image.size
            mode = image.mode

            camera_dict[cam_type] = {
                'cam_path': cam_path,
                'cam2lidar_mat': cam2lidar_mat,
                'cam_intrinsic': cam_intrinsic,
                'cam_token': cam_token,
                'image': image,
                'w': size_w, 
                'h': size_h,
                'ori_w': size_w, 
                'ori_h': size_h,
                'mode': mode,
                'rotation': np.eye(2),
                'translation': np.zeros(2),
            }
        return camera_dict      

    def match_lidar2image(self, camera_dict, pcds, pcds_label, use_consistency=False):
        rand_level = self.rand_level
        point_num = pcds.shape[0]
        pcds_data = np.concatenate((pcds[:, :3].T, np.ones((1, point_num))))
        
        point_coord = []
        point_mask = []
        images = []  # batch！
        camera_list = []

        point_coord2 = []
        point_mask2 = []
        
        for cam_type, cam_info in camera_dict.items():
            camera_list.append(cam_type)
            img = cam_info['image']
            images.append(img)
            pc = copy.deepcopy(pcds_data)
            
            cam2lidar_mat = cam_info['cam2lidar_mat']
            cam_intrinsic = cam_info['cam_intrinsic']

            if use_consistency:
                pc2 = copy.deepcopy(pcds_data)
                if rand_level < 0:
                    theta = random.uniform(-2, 2)*math.pi/180
                    rotx_matrix = np.asarray([[1.0, 0.0, 0.0],
                                              [0.0, math.cos(theta), -math.sin(theta)],
                                              [0.0, math.sin(theta), math.cos(theta)]])
                    theta = random.uniform(-2, 2)*math.pi/180
                    roty_matrix = np.asarray([[math.cos(theta), 0.0, math.sin(theta)],
                                              [0.0, 1.0, 0.0],
                                              [-math.sin(theta), 0.0, math.cos(theta)]])

                    theta = random.uniform(-2, 2)*math.pi/180
                    rotz_matrix = np.asarray([[math.cos(theta), -math.sin(theta), 0.0],
                                             [math.sin(theta), math.cos(theta), 0.0],
                                             [0.0, 0.0, 1.0]])
                    rot_mat = (rotz_matrix @ roty_matrix @ rotx_matrix)
                    pc2[:3, :] = np.dot(rot_mat, pc2[:3, :])

                pc_camera2 = np.dot(cam2lidar_mat, pc2)
                depths2 = pc_camera2[2,:]
                points2 = view_points(pc_camera2[:3, :], np.array(cam_intrinsic), normalize=True)

                ori_w, ori_h = cam_info['ori_w'], cam_info['ori_h'] 
                mask21 = np.ones(depths2.shape[0], dtype=bool)
                mask21 = np.logical_and(mask21, depths2 > 1)
                mask21 = np.logical_and(mask21, points2[0, :] > 1)
                mask21 = np.logical_and(mask21, points2[0, :] < ori_w - 1)
                mask21 = np.logical_and(mask21, points2[1, :] > 1)
                mask21 = np.logical_and(mask21, points2[1, :] < ori_h - 1)

                w, h = cam_info['w'], cam_info['h']
                # image aug matrix
                img_aug_matrix = np.eye(4)
                img_aug_matrix[:2, :2] = cam_info['rotation'] 
                img_aug_matrix[:2, 3] = cam_info['translation']
                points2 = np.dot(img_aug_matrix[:3, :3], points2)
                points2 += img_aug_matrix[:3, 3].reshape(-1, 1)

                mask22 = np.ones(depths2.shape[0], dtype=bool)
                mask22 = np.logical_and(mask22, depths2 > 1)
                mask22 = np.logical_and(mask22, points2[0, :] > 1)
                mask22 = np.logical_and(mask22, points2[0, :] < w - 1)
                mask22 = np.logical_and(mask22, points2[1, :] > 1)
                mask22 = np.logical_and(mask22, points2[1, :] < h - 1)

                mask23 = np.logical_and(mask21, mask22)
                points2 = points2.T
                points2[~mask23] = points2[~mask23]*0-10.0

                point_coord2.append(points2[:,:2].astype(np.float32))
                point_mask2.append(mask23)

            # random rotation
            if rand_level >= 1:
                theta = random.uniform(-rand_level, rand_level)*math.pi/180
                rotx_matrix = np.asarray([[1.0, 0.0, 0.0],
                                          [0.0, math.cos(theta), -math.sin(theta)],
                                          [0.0, math.sin(theta), math.cos(theta)]])

                theta = random.uniform(-rand_level, rand_level)*math.pi/180
                roty_matrix = np.asarray([[math.cos(theta), 0.0, math.sin(theta)],
                                          [0.0, 1.0, 0.0],
                                          [-math.sin(theta), 0.0, math.cos(theta)]])

                theta = random.uniform(-rand_level, rand_level)*math.pi/180
                rotz_matrix = np.asarray([[math.cos(theta), -math.sin(theta), 0.0],
                                         [math.sin(theta), math.cos(theta), 0.0],
                                         [0.0, 0.0, 1.0]])
        
                rot_mat = (rotz_matrix @ roty_matrix @ rotx_matrix)
                pc[:3, :] = np.dot(rot_mat, pc[:3, :])

            pc_camera = np.dot(cam2lidar_mat, pc)
            depths = pc_camera[2,:]
            points = view_points(pc_camera[:3, :], np.array(cam_intrinsic), normalize=True)
            ego_dist = 1

            ori_w, ori_h = cam_info['ori_w'], cam_info['ori_h']  # 1600, 900
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > 1)
            mask = np.logical_and(mask, points[0, :] > 1)
            mask = np.logical_and(mask, points[0, :] < ori_w - 1)
            mask = np.logical_and(mask, points[1, :] > 1)
            mask = np.logical_and(mask, points[1, :] < ori_h - 1)

            w, h = cam_info['w'], cam_info['h']
            # image aug matrix
            img_aug_matrix = np.eye(4)
            img_aug_matrix[:2, :2] = cam_info['rotation'] 
            img_aug_matrix[:2, 3] = cam_info['translation']
            points = np.dot(img_aug_matrix[:3, :3], points)
            points += img_aug_matrix[:3, 3].reshape(-1, 1)

            mask2 = np.ones(depths.shape[0], dtype=bool)
            mask2 = np.logical_and(mask2, depths > ego_dist)
            mask2 = np.logical_and(mask2, points[0, :] > 1)
            mask2 = np.logical_and(mask2, points[0, :] < w - 1)
            mask2 = np.logical_and(mask2, points[1, :] > 1)
            mask2 = np.logical_and(mask2, points[1, :] < h - 1)

            mask = np.logical_and(mask, mask2)

            points = points.T
            points[~mask] = points[~mask]*0-10.0

            points_int = points.astype(np.int32)
            points_int = points_int[:,:2]
            # image_label = np.zeros((w, h), dtype=pcds_label.dtype)  # init with 0 ?
            # image_label[points_int[mask][:,0], points_int[mask][:,1]] = pcds_label[mask]
            # image_labels.append(image_label)
            
            assert points_int.shape[0] == point_num
            assert mask.shape[0] ==  point_num
            
            point_coord.append(points[:,:2].astype(np.float32))
            point_mask.append(mask)
            
        images = np.asarray(images, dtype=np.float32)  # [6, W, H, 3]
        point_coord = np.asarray(point_coord, dtype=np.float32) # [6, N, 2]
        point_mask = np.asarray(point_mask, dtype=np.bool) # [6, N]
        # image_labels = np.asarray(image_labels, dtype=np.int32) # [6, W, H]

        camera_input = {
            'images': images,
            'point_coord': point_coord,
            'point_mask': point_mask,
        }
        if use_consistency:
            point_coord2 = np.asarray(point_coord2, dtype=np.float32) # [6, N, 2]
            point_mask2 = np.asarray(point_mask2, dtype=np.bool) # [6, N]
            camera_input.update({
                'point_coord2': point_coord2,
                'point_mask2': point_mask2,
            })
        return camera_input
    
    def cam2lidar_aug(self, camera_dict):
        for cam_type, cam_info in camera_dict.items():
            cam_info = self.image_aug(cam_info)
            camera_dict[cam_type] = cam_info

        return camera_dict

    def resample_points(self, points_num, point_sample_num, pcds_label_use=None):
        # resample
        if points_num >= point_sample_num:
            choice = np.random.choice(points_num, point_sample_num, replace=False)
        else:
            idx1 = np.arange(0,points_num,1)
            idx2 = np.random.choice(points_num, point_sample_num-points_num, replace=True)
            choice = np.concatenate([idx1, idx2])
        return choice

    def reset_sample(self, is_aug=True):
        self.frame_point_num = random.choice(self.config.frame_point_num)
        if not is_aug:
            self.point_aug = None
            self.cp_aug = None
            self.rand_level = 0

    def form_batch(self, pcds_total, gt_boxes=None, no_aug=False):
        """
        pcds_extra: feature from extra infomation, N x D
        """
        # point aug
        if (not no_aug) and (self.point_aug is not None):
            pcds_total, gt_boxes = self.point_aug(pcds_total, gt_boxes)

        pcds_xyzi = pcds_total[:, :4]
        pcds_target = pcds_total[:, 4]
        pcds_extra = pcds_total[:, 5:]

        batch_data = dict()
        bev_coord = utils.Quantize(pcds_xyzi,**self.config.Voxel.bev_params)
        rv_coord = utils.SphereQuantize(pcds_xyzi,**self.config.Voxel.rv_params)
            
        pcds_xyzi = utils.make_point_feat(pcds_xyzi, bev_coord, rv_coord)
        pcds_xyzi = np.concatenate((pcds_xyzi, pcds_extra), axis=1)

        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32))
        pcds_xyzi = pcds_xyzi.transpose(1, 0).contiguous()

        bev_coord = torch.FloatTensor(bev_coord.astype(np.float32))
        rv_coord = torch.FloatTensor(rv_coord.astype(np.float32))

        pcds_target = torch.LongTensor(pcds_target.astype(np.long))

        batch_data.update({
            'pcds_xyzi': pcds_xyzi.unsqueeze(-1),
            'bev_coord': bev_coord.unsqueeze(-1),
            'rv_coord': rv_coord.unsqueeze(-1),
            'pcds_target': pcds_target.unsqueeze(-1),
            'gt_boxes': gt_boxes
        })
        return batch_data

    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, idx):
        info = self.data_infos[idx]
        lidar_token = info['lidar_token']
        sample_token = info['token']
        lidar_path = info['lidar_path']
        
        pcds = self.load_pcd_from_file(lidar_path)
        pcds_label, gt_boxes = self.load_lidar_anno(info)

        # copy-paste augmentation
        pcds_idxs = None
        if self.cp_aug is not None:
            pcds, pcds_label, gt_boxes, pcds_idxs = self.cp_aug(pcds, pcds_label, gt_3d_box=gt_boxes)

        # extra feats
        pcds_toal_feats = [pcds, pcds_label[:, np.newaxis]]

        # load camera feat
        if self.use_camera == 'camera_raw':
            cam_infos = info['cams']
            camera_dict = self.load_camera_image(cam_infos)
            camera_dict = self.cam2lidar_aug(camera_dict)
            camera_input = self.match_lidar2image(camera_dict, pcds, pcds_label, self.config.use_consistency)
        else:
            pass

        pcds_total = np.concatenate(pcds_toal_feats, axis=1)

        # resample points
        choice_idxs = self.resample_points(pcds.shape[0], self.frame_point_num, pcds_label)
        pcds_total = pcds_total[choice_idxs]
        point_sample_num = pcds_total.shape[0]

        data_dic = {
            'sample_token': sample_token,
            'lidar_token': lidar_token,
            'point_sample_num': point_sample_num
        }

        if self.use_camera == 'camera_raw':
            point_coord = camera_input['point_coord']
            point_coord = point_coord[:, choice_idxs]

            point_mask = camera_input['point_mask']
            point_mask = point_mask[:, choice_idxs]

            point_image = camera_input['images']

            data_dic.update({
                'images':torch.FloatTensor(point_image),
                'point_coord':torch.FloatTensor(point_coord),
                'point_mask':torch.BoolTensor(point_mask),
            })
            if self.config.use_consistency:
                point_coord2 = camera_input['point_coord2']
                point_coord2 = point_coord2[:, choice_idxs]
                point_mask2 = camera_input['point_mask2']
                point_mask2 = point_mask2[:, choice_idxs]
                data_dic.update({
                    'point_coord_raw':torch.FloatTensor(point_coord2),
                    'point_mask_raw':torch.BoolTensor(point_mask2),
                })

        # preprocess
        batch_data = self.form_batch(pcds_total.copy(), gt_boxes)
        data_dic.update({
            'pcds_xyzi': batch_data['pcds_xyzi'],
            'pcds_coord': batch_data['bev_coord'],
            'pcds_sphere_coord': batch_data['rv_coord'],
            'pcds_target': batch_data['pcds_target'],
        })
        if self.config.use_consistency:
            batch_data_raw = self.form_batch(pcds_total.copy(), gt_boxes, no_aug=True)
            data_dic.update({
                'pcds_xyzi_raw': batch_data_raw['pcds_xyzi'],
                'pcds_coord_raw': batch_data_raw['bev_coord'],
                'pcds_sphere_coord_raw': batch_data_raw['rv_coord'],
            })

        return data_dic
    

# define the class of dataloader
class DataloadVal(DataloadTrain):
    def __init__(self, config):
        super(DataloadVal, self).__init__(config)

    def reset_sample(self):
        self.point_aug = None
        self.cp_aug = None
        self.rand_level = 0
        
    def load_infos(self, info_path):
        print('load data infos!')
        with open(info_path, 'rb') as f:
            self.data_infos = pkl.load(f)['infos']
        self.sample_length = len(self.data_infos)
        print('{} Samples: '.format(self.mode), self.sample_length)

    def form_batch(self, pcds_total):
        """
        pcds_extra: feature from extra infomation, N x D
        """
        # point aug

        pcds_xyzi = pcds_total[:, :4]
        pcds_target = pcds_total[:, 4]
        pcds_extra = pcds_total[:, 5:]

        bev_coord = utils.Quantize(pcds_xyzi,**self.config.Voxel.bev_params)
        rv_coord = utils.SphereQuantize(pcds_xyzi,**self.config.Voxel.rv_params)
        pcds_xyzi = utils.make_point_feat(pcds_xyzi, bev_coord, rv_coord)

        pcds_xyzi = np.concatenate((pcds_xyzi, pcds_extra), axis=1)

        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32))
        pcds_xyzi = pcds_xyzi.transpose(1, 0).contiguous()

        bev_coord = torch.FloatTensor(bev_coord.astype(np.float32))
        rv_coord = torch.FloatTensor(rv_coord.astype(np.float32))

        pcds_target = torch.LongTensor(pcds_target.astype(np.long))

        return pcds_xyzi.unsqueeze(-1), bev_coord.unsqueeze(-1), rv_coord.unsqueeze(-1), pcds_target.unsqueeze(-1)

    def __getitem__(self, idx):
        info = self.data_infos[idx]

        lidar_path = info['lidar_path']
        lidarseg_path = info['lidarseg_path']
        lidar_token = info['lidar_token']
        sample_token = info['token']

        pcds = self.load_pcd_from_file(lidar_path)
        pcds_label, gt_boxes = self.load_lidar_anno(info)
        
        # extra feats
        pcds_toal_feats = [pcds, pcds_label[:, np.newaxis]]
        point_sample_num = pcds.shape[0]

        # load camera raw data
        if self.use_camera == 'camera_raw':
            cam_infos = info['cams']
            camera_dict = self.load_camera_image(cam_infos)
            camera_dict = self.cam2lidar_aug(camera_dict)
            camera_input = self.match_lidar2image(camera_dict, pcds, pcds_label)

        else:
            pass

        pcds_total = np.concatenate(pcds_toal_feats, axis=1)

        # data aug
        pcds_xyzi_list = []
        pcds_coord_list = []
        pcds_sphere_coord_list = []
        pcds_target_list = []

        if self.config.test_flip:
            for x_sign in [1, -1]:
                for y_sign in [1, -1]:
                    pcds_tmp = pcds_total.copy()
                    pcds_tmp[:, 0] *= x_sign
                    pcds_tmp[:, 1] *= y_sign
                    pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target = self.form_batch(pcds_tmp)

                    pcds_xyzi_list.append(pcds_xyzi)
                    pcds_coord_list.append(pcds_coord)
                    pcds_sphere_coord_list.append(pcds_sphere_coord)
                    pcds_target_list.append(pcds_target)
        else:
            pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target = self.form_batch(pcds_total)
            pcds_xyzi_list.append(pcds_xyzi)
            pcds_coord_list.append(pcds_coord)
            pcds_sphere_coord_list.append(pcds_sphere_coord)
            pcds_target_list.append(pcds_target)
        
        pcds_xyzi = torch.stack(pcds_xyzi_list, dim=0)
        pcds_coord = torch.stack(pcds_coord_list, dim=0)
        pcds_sphere_coord = torch.stack(pcds_sphere_coord_list, dim=0)
        pcds_target = torch.stack(pcds_target_list, dim=0)

        # pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target = self.form_batch(pcds_total)

        data_dic = {
            'pcds_xyzi': pcds_xyzi,
            'pcds_coord': pcds_coord,
            'pcds_sphere_coord': pcds_sphere_coord,
            'pcds_target': pcds_target,
            'sample_token': sample_token,
            'lidar_token': lidar_token,
            'point_sample_num': point_sample_num,
        }

        if self.use_camera == 'camera_raw':
            data_dic.update({
                'images':torch.FloatTensor(camera_input['images']),
                'point_coord':torch.FloatTensor(camera_input['point_coord']),
                'point_mask':torch.BoolTensor(camera_input['point_mask']),
            })

        return data_dic

    def __len__(self):
        return len(self.data_infos)