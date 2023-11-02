import os
import yaml
import pickle as pkl
import numpy as np
from nuscenes import NuScenes

import utils
from multi_process import MultiProcess

import pdb

fpath_nusc = '../../nuscenes'
fpath_nusc_bank = os.path.join(fpath_nusc, 'nusc_bank')
class_set = set(('barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck'))

for cate in class_set:
    fpath_cate = os.path.join(fpath_nusc_bank, cate)
    os.system("mkdir -p {}".format(fpath_cate))


with open('../../datasets/nusc/nuscenes.yaml', 'r') as f:
    task_cfg = yaml.load(f)


common_infos=dict(class_set=class_set, fpath_nusc_bank=fpath_nusc_bank, task_cfg=task_cfg)
nusc = NuScenes(version='v1.0-trainval', dataroot=fpath_nusc, verbose=True)

flist = []
with open(os.path.join(nusc.dataroot, 'nutcp_infos_train.pkl'), 'rb') as f:
    data_infos = pkl.load(f)['infos']
    for info in data_infos:
        lidar_sd_token = nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        gt_names = info['gt_names']
        gt_boxes = info['gt_boxes']
        fname_label = os.path.join(nusc.dataroot, nusc.get('lidarseg', lidar_sd_token)['filename'])
        fname_lidar = os.path.join(nusc.dataroot, '{}/{}/{}'.format(*info['lidar_path'].split('/')[-3:]))
        assert gt_names.shape[0] == gt_boxes.shape[0]
        if gt_names.shape[0] > 0:
            gt_boxes[:, 6] *= -1
            flist.append((gt_names, gt_boxes, fname_label, fname_lidar))


# multi-process
multi_obj = MultiProcess(flist, utils.process_single, num_workers=40, common_infos=common_infos)
for i, data in enumerate(multi_obj.run()):
    if i % 5000 == 0:
        print("Samples: {}".format(i))
