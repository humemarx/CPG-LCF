import numpy as np
import random
import yaml
import os
import cv2
from scipy.spatial import Delaunay

import pdb

def in_range(v, r):
    return (v >= r[0]) * (v < r[1])


def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0

def compute_box_3d(center, size, yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    
    # 3d bounding box dimensions
    l = size[0]
    w = size[1]
    h = size[2]
    
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    return corners_3d.T

def csr2corners(center, size, yaw):
    '''
        0 -------- 1
       /|         /|
      3 -------- 2 .
      | |        | |
      . 4 -------- 5
      |/         |/
      7 -------- 6
    Input: 
        center, size, 3
        yaw, 1
    Output:
        corners_3d, (8, 3)
    '''
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    
    # 3d bounding box dimensions
    l = size[0]
    w = size[1]
    h = size[2]
    
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.stack((x_corners, y_corners, z_corners), axis=0))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = corners_3d.T
    return corners_3d

def csr2corners_batch(gt_3d_box):
    '''
    Input:
        gt_3d_box, (N, 7), 7 -> (cx, cy, cz, l, w, h, yaw)
    Output:
        gt_3d_box_corners, (N, 8, 3)
    '''
    gt_3d_box_corners = []
    for i in range(gt_3d_box.shape[0]):
        corners_3d_tmp = csr2corners(center=gt_3d_box[i, :3], size=gt_3d_box[i, 3:6], yaw=gt_3d_box[i, 6])
        gt_3d_box_corners.append(corners_3d_tmp)
    
    gt_3d_box_corners = np.stack(gt_3d_box_corners, axis=0)
    return gt_3d_box_corners

def corners2csr_batch(gt_3d_box_corners):
    '''
        0 -------- 1
       /|         /|
      3 -------- 2 .
      | |        | |
      . 4 -------- 5
      |/         |/
      7 -------- 6
    Input:
        gt_3d_box_corners, (N, 8, 3)
    Output:
        gt_3d_box, (N, 7), 7 -> (cx, cy, cz, l, w, h, yaw)
    '''
    center = gt_3d_box_corners.mean(axis=1) #(N, 3)
    l = np.sqrt(np.power(gt_3d_box_corners[:, [0, 1, 4, 5]] - gt_3d_box_corners[:, [3, 2, 7, 6]], 2).sum(axis=2)).mean(axis=1, keepdims=True)
    w = np.sqrt(np.power(gt_3d_box_corners[:, [0, 3, 4, 7]] - gt_3d_box_corners[:, [1, 2, 5, 6]], 2).sum(axis=2)).mean(axis=1, keepdims=True)
    h = np.sqrt(np.power(gt_3d_box_corners[:, [0, 1, 2, 3]] - gt_3d_box_corners[:, [4, 5, 6, 7]], 2).sum(axis=2)).mean(axis=1, keepdims=True)
    r_tmp = gt_3d_box_corners[:, [0, 1, 4, 5], :2] - gt_3d_box_corners[:, [3, 2, 7, 6], :2]
    r = np.arctan2(r_tmp[:, :, 1], r_tmp[:, :, 0]).mean(axis=1, keepdims=True)
    gt_3d_box = np.concatenate((center, l, w, h, r), axis=1)
    return gt_3d_box

def rotate_along_z(pcds, theta):
    rotateMatrix = cv2.getRotationMatrix2D((0, 0), theta, 1.0)[:, :2].T
    pcds[:, :2] = pcds[:, :2].dot(rotateMatrix)
    return pcds


def random_f(r):
    return r[0] + (r[1] - r[0]) * random.random()


class CutPaste:
    def __init__(self, config):
        self.object_dir = config.ObjBankDir
        if isinstance(config.category_list, list):
            self.sub_dirs = config.category_list
            self.class_list = self.sub_dirs
        elif isinstance(config.category_list, dict):
            self.sub_dirs = []
            self.class_list = []
            for key, value in config.category_list.items():
                self.sub_dirs.append(key)
                self.class_list += [key]*value
        else:
            self.sub_dirs = config.category_list
            self.class_list = self.sub_dirs

        if hasattr(config, 'idmap'):
            self.idmap = config.idmap
        else:
            self.idmap = {}

        self.sub_dirs_dic = {}
        for fp in self.sub_dirs:
            fpath = os.path.join(self.object_dir, fp)
            fname_list = [os.path.join(fpath, x) for x in os.listdir(fpath) if x.endswith('.npz')]
            print('Load {0}: {1}'.format(fp, len(fname_list)))
            self.sub_dirs_dic[fp] = fname_list
        
        self.paste_max_obj_num = config.paste_max_obj_num
        self.road_idx = config.road_idx
        self.things_label_range = config.things_label_range
        self.min_pts_num = config.min_pts_num
    
    def get_random_rotate_along_z_obj(self, pcds_obj, bbox_corners, theta):
        pcds_obj_result = rotate_along_z(pcds_obj, theta)
        bbox_corners_result = rotate_along_z(bbox_corners, theta)
        return pcds_obj_result, bbox_corners_result
    
    def get_fov(self, pcds_obj):
        x, y, z = pcds_obj[:, 0], pcds_obj[:, 1], pcds_obj[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12
        u = np.sqrt(x ** 2 + y ** 2) + 1e-12

        phi = np.arctan2(x, y)
        theta = np.arcsin(z / d)

        u_fov = (u.min(), u.max())
        phi_fov = (phi.min(), phi.max())
        theta_fov = (theta.min(), theta.max())
        return u_fov, phi_fov, theta_fov
    
    def no_occlusion_check(self, pcds, pcds_label, phi_fov, theta_fov):
        x, y, z = pcds[:, 0], pcds[:, 1], pcds[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12
        u = np.sqrt(x ** 2 + y ** 2) + 1e-12

        phi = np.arctan2(x, y)
        theta = np.arcsin(z / d)

        fov_mask = in_range(phi, phi_fov) * in_range(theta, theta_fov)
        pcds_label_in_fov = pcds_label[fov_mask]
        in_fov_obj_mask = (pcds_label_in_fov >= self.things_label_range[0]) * (pcds_label_in_fov <= self.things_label_range[1])
        if in_fov_obj_mask.sum() < 3:
            return True, fov_mask
        else:
            return False, fov_mask
    
    def no_collision_check(self, pcds, pcds_label, bbox_corners):
        in_box3d_mask = in_hull(pcds[:, :3], bbox_corners)
        pcds_label_in_box = pcds_label[in_box3d_mask]
        in_box_obj_mask = (pcds_label_in_box >= self.things_label_range[0]) * (pcds_label_in_box <= self.things_label_range[1])
        if in_box_obj_mask.sum() < 3:
            return True
        else:
            return False
    
    def paste_single_obj(self, pcds, pcds_road, pcds_label, idx_mask, gt_3d_box=None):
        '''
        Input:
            pcds, (N, 4), 4 -> x, y, z, intensity
            pcds_road, (M, 4)
            pcds_label, (N,)
        Output:
            pcds, (N1, 4)
            pcds_label, (N1,)
        '''
        # pcds (N, 4), 4 contains x, y, z, intensity
        # pcds_label(N)
        cate = random.choice(self.class_list)
        fname_npz = random.choice(self.sub_dirs_dic[cate])
        npkl = np.load(fname_npz)

        pcds_obj = npkl['pcds']
        cate_id = int(npkl['cate_id'])
        semantic_cate = str(npkl['cate'])
        if semantic_cate in self.idmap:
            cate_id = self.idmap[semantic_cate]

        if 'gt_3d_box' in npkl:
            obj_3d_box = npkl['gt_3d_box']
            bbox_corners = csr2corners(obj_3d_box[:3], obj_3d_box[3:6], obj_3d_box[6])
        else:
            bbox_center = npkl['center']
            bbox_size = npkl['size'] * 1.05
            bbox_yaw = npkl['yaw']
            bbox_corners = compute_box_3d(bbox_center, bbox_size, bbox_yaw)

        if(len(pcds_obj) < self.min_pts_num):
            return pcds, pcds_label, idx_mask, gt_3d_box, 0
        
        theta_list = np.arange(0, 360, 18).tolist()
        np.random.shuffle(theta_list)
        for theta in theta_list:
            # global rotate object
            pcds_obj_aug, bbox_corners_aug = self.get_random_rotate_along_z_obj(pcds_obj, bbox_corners, theta)

            # get local road height
            valid_road_mask = in_hull(pcds_road[:, :2], bbox_corners_aug[:4, :2])
            pcds_local_road = pcds_road[valid_road_mask]
            if pcds_local_road.shape[0] > 5:
                road_mean_height = float(pcds_local_road[:, 2].mean())
                z_shift_value = road_mean_height - bbox_corners_aug[:, 2].min()
                pcds_obj_aug[:, 2] += z_shift_value
            else:
                # object is not on road
                continue
            
            # get object fov
            u_fov, phi_fov, theta_fov = self.get_fov(pcds_obj_aug)
            if (abs(u_fov[1] - u_fov[0]) < 8) and (abs(phi_fov[1] - phi_fov[0]) < 1) and (abs(theta_fov[1] - theta_fov[0]) < 1):
                # if it is occlusion with the existing objects
                no_occlusion_flag, fov_mask = self.no_occlusion_check(pcds, pcds_label, phi_fov, theta_fov)
                # if it collides with existing objects
                no_collision_flag = self.no_collision_check(pcds, pcds_label, bbox_corners_aug)

                if no_occlusion_flag and no_collision_flag:
                    assert pcds.shape[0] == pcds_label.shape[0]

                    # add object back
                    pcds_filter = pcds[~fov_mask]
                    pcds_label_filter = pcds_label[~fov_mask]
                    idx_mask = idx_mask[~fov_mask[:idx_mask.shape[0]]]

                    pcds = np.concatenate((pcds_filter, pcds_obj_aug), axis=0)

                    pcds_addobj_label = np.full((pcds_obj_aug.shape[0],), fill_value=cate_id, dtype=pcds_label.dtype)
                    pcds_label = np.concatenate((pcds_label_filter, pcds_addobj_label), axis=0)

                    if gt_3d_box is not None:
                        gt_3d_box = np.pad(gt_3d_box, ((0,1),(0,0)), 'constant', constant_values=0)
                        obj_box_aug = corners2csr_batch(bbox_corners_aug.reshape(1, 8, 3))
                        gt_3d_box[-1, :7] = obj_box_aug[0]
                        gt_3d_box[-1, 7] = cate_id

                    break
                else:
                    # invalid heading
                    continue
            else:
                break
        
        return pcds, pcds_label, idx_mask, gt_3d_box, 1
    
    def __call__(self, pcds, pcds_label, pcds_road_label=None, gt_3d_box=None):
        '''
        Input:
            pcds, (N, 4), 4 -> x, y, z, intensity
            pcds_label, (N,)
            gt_3d_box, (K, 8), 8 -> (cx, cy, cz, sx, sy, sz, yaw, cls)
        '''
        idx_mask = np.arange(0, pcds.shape[0], 1)
        paste_obj_num = random.randint(0, self.paste_max_obj_num)
        if paste_obj_num == 0:
            return pcds, pcds_label, gt_3d_box, idx_mask
        else:
            if pcds_road_label is not None:
                pcds_road = [pcds[pcds_road_label == i] for i in self.road_idx]
            else:
                pcds_road = [pcds[pcds_label == i] for i in self.road_idx]
            pcds_road = np.concatenate(pcds_road, axis=0)

            pcds_new = pcds.copy()
            pcds_label_new = pcds_label.copy()
            if gt_3d_box is not None:
                gt_3d_box_new = gt_3d_box.copy()
            else:
                gt_3d_box_new = None

            paste_num = 0
            while paste_num <= paste_obj_num:
                pcds_new, pcds_label_new, idx_mask, gt_3d_box_new, paste_flag = self.paste_single_obj(pcds_new, pcds_road, pcds_label_new, idx_mask, gt_3d_box_new)
                paste_num += paste_flag

            return pcds_new, pcds_label_new, gt_3d_box_new, idx_mask



if __name__ == '__main__':
    def relabel(pcds_labels, label_map):
        result_labels = np.zeros((pcds_labels.shape[0],), dtype=pcds_labels.dtype)
        for key in label_map:
            value = label_map[key]
            mask = (pcds_labels == key)
            result_labels[mask] = value
        return result_labels

    class CopyPasteAug:
        is_use = False
        ObjBankDir = './data/waymo2/obj_bank'
        paste_max_obj_num = 50
        category_list = ('VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST')
        road_idx = [17, 18, 19, 20, 22]
        things_label_range = [1, 13]
        min_pts_num = 50
        idmap = {
            "VEHICLE":4,
            "PEDESTRIAN":7,
            "SIGN":8,
            "CYCLIST":6
        }

    with open('datasets/waymo/waymo.yaml', 'r') as f:
        task_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fname_path = './data/waymo2/training/samples/LIDAR_TOP/15832924468527961_1564_160_1584_160_1507678829280022.npz'

    config = CopyPasteAug
    cp_aug = CutPaste(config)

    pcd_data=np.load(fname_path)
    pcds_xyzi = pcd_data['lidar_pcds']
    pcds_label = pcd_data['lidar_label']

    pcds, pcds_label_use, _, _ = cp_aug(pcds_xyzi, pcds_label)

    fname_path2 = '15832924468527961_1564_160_1584_160_1507678829280022_cp.npz'
    np.savez(fname_path2, lidar_pcds=pcds, lidar_label=pcds_label_use)