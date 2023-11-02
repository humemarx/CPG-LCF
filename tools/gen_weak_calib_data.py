# coding=utf-8
'''
Author: husserl
License: Apache Licence
Software: VSCode
Date: 2023-07-20 08:59:20
LastEditors: husserl
LastEditTime: 2023-10-25 11:14:57
'''
import pickle as pkl
import random
import numpy as np
import math
import os

if __name__ == '__main__':
    # random.seed(42)
    SeqDir = 'nuscenes'
    info_path = os.path.join(SeqDir, 'nutcp_infos_val.pkl')
    rand_level = 0
    new_info_path = os.path.join(SeqDir, 'nutcp_infos_val{}.pkl'.format(rand_level))

    with open(info_path, 'rb') as f:
        data_infos = pkl.load(f)['infos']

    pc = np.random.rand(3,2)

    for idx, info in enumerate(data_infos):
        cam_infos = info['cams']

        for cam_type, cam_info in cam_infos.items():
            cam2lidar_mat = cam_info['sensor2lidar_mat']
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

            pc1 = np.dot(rot_mat, pc)
            pc_camera1 = np.dot(cam2lidar_mat, np.concatenate((pc1, np.ones((1, 2)))))

            rot_mat1 = np.concatenate((np.dot(cam2lidar_mat[:, :3], rot_mat), cam2lidar_mat[:, 3:]), axis=-1)
            pc_camera2 = np.dot(rot_mat1, np.concatenate((pc, np.ones((1, 2)))))

            # print(cam2lidar_mat)
            # print(rot_mat1)
            
            diff = np.max(np.abs(pc_camera1-pc_camera2))
            if diff > 1e-10:
                print(diff)

            cam_info['sensor2lidar_mat'] = np.concatenate((np.dot(cam2lidar_mat[:, :3], rot_mat), cam2lidar_mat[:, 3:]), axis=-1)
            cam_infos[cam_type] = cam_info
    
        info['cams'] = cam_infos
        data_infos[idx] = info

    data_pkl = {'infos': data_infos}

    with open(new_info_path, 'wb') as f:
        pkl.dump(data_pkl, f)


    