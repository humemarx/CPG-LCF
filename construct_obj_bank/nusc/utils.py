import numpy as np
import random
import yaml
import os
import cv2
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

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


def rotate_along_z(pcds, theta):
    rotateMatrix = cv2.getRotationMatrix2D((0, 0), theta, 1.0)[:, :2].T
    pcds[:, :2] = pcds[:, :2].dot(rotateMatrix)
    return pcds


def random_f(r):
    return r[0] + (r[1] - r[0]) * random.random()


def relabel(pcds_labels, label_map):
    result_labels = np.zeros((pcds_labels.shape[0],), dtype=pcds_labels.dtype)
    for key in label_map:
        value = label_map[key]
        mask = (pcds_labels == key)
        result_labels[mask] = value
    
    return result_labels


def compute_3dbox_iou(box1_corners, box2_corners):
    '''
    Input:
        box1_corners, box2_corners, (8, 3)
    '''
    box1_bev_poly = Polygon(box1_corners[[0,1,2,3], :2])
    box2_bev_poly = Polygon(box2_corners[[0,1,2,3], :2])

    # bev IoU
    box_12_bev_intersec = box1_bev_poly.intersection(box2_bev_poly).area

    # 3D IoU
    box_12_h_intersec = min(box1_corners[0, 2], box2_corners[0, 2]) - max(box1_corners[7, 2], box2_corners[7, 2])
    box_12_h_intersec = max(0, box_12_h_intersec)
    box1_h = box1_corners[0, 2] - box1_corners[7, 2]
    box2_h = box2_corners[0, 2] - box2_corners[7, 2]
    box_3d_intersec = box_12_bev_intersec * box_12_h_intersec
    iou_3d = box_3d_intersec / (box1_bev_poly.area * box1_h + box2_bev_poly.area * box2_h - box_3d_intersec + 1e-12)
    return iou_3d


def compute_3dbox_iou_self(box_corners):
    '''
    Input:
        box_corners, (N, 8, 3)
    '''
    N = len(box_corners)
    iou_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i+1, N):
            iou_mat[i, j] = compute_3dbox_iou(box_corners[i], box_corners[j])
    return iou_mat


def process_single(meta_data, common_infos=None):
    gt_names, gt_boxes, fname_label, fname_lidar = meta_data

    class_set = common_infos['class_set']
    fpath_nusc_bank = common_infos['fpath_nusc_bank']
    task_cfg = common_infos['task_cfg']

    pcds_xyzi = np.fromfile(fname_lidar, dtype=np.float32, count=-1).reshape((-1, 5))[:, :4]
    pcds_label = np.fromfile(fname_label, dtype=np.uint8).reshape((-1))
    pcds_label_use = relabel(pcds_label, task_cfg['learning_map'])

    gt_boxes_corners = [compute_box_3d(gt_boxes[i, :3], gt_boxes[i, 3:6], gt_boxes[i, 6]) for i in range(gt_boxes.shape[0])]
    gt_boxes_corners = np.stack(gt_boxes_corners, axis=0)
    iou_mat = compute_3dbox_iou_self(gt_boxes_corners)
    delete_index_set = set(np.where(iou_mat > 0)[0]) | set(np.where(iou_mat > 0)[1])
    for i in range(gt_names.shape[0]):
        if (i not in delete_index_set) and (gt_names[i] in class_set):
            obj_mask = in_hull(pcds_xyzi[:, :3], gt_boxes_corners[i])
            pcds_obj = pcds_xyzi[obj_mask]
            pcds_label_use_obj = pcds_label_use[obj_mask]

            if pcds_obj.shape[0] > 0:
                label_list = np.unique(pcds_label_use_obj).tolist()
                for label in label_list:
                    label_name = task_cfg['labels_16'][label]
                    if label_name == gt_names[i]:
                        obj_refine_mask = (pcds_label_use_obj == label)
                        pcds_obj_refine = pcds_obj[obj_refine_mask]
                        pcds_label_use_obj_refine = pcds_label_use_obj[obj_refine_mask]
                        if pcds_obj_refine.shape[0] > 5:
                            fname_obj = os.path.join(fpath_nusc_bank, label_name, "{}###{}.npz".format(os.path.basename(fname_lidar), i))
                            np.savez_compressed(fname_obj, pcds=pcds_obj_refine, cate_id=label, cate=label_name,
                            center=gt_boxes[i, :3], size=gt_boxes[i, 3:6], yaw=gt_boxes[i, 6])

                            yield 1
