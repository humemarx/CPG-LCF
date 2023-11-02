# coding=utf-8
'''
Author: husserl
License: Apache Licence
Software: VSCode
Date: 2023-03-01 08:24:39
LastEditors: husserl
LastEditTime: 2023-07-04 12:15:53
'''
from typing import Any
import numpy as np
import random
import math
import torch
from typing import List, Optional, Sequence, Tuple, Union
import numba

@numba.jit
def points_in_convex_polygon_jit(points, polygon, clockwise=True):
    """check points is in 2d convex polygons. True when point in polygon
    Args:
        points: [num_points, 2] array.
        polygon: [num_polygon, num_points_of_polygon, 2] array.
        clockwise: bool. indicate polygon is clockwise.
    Returns:
        [num_points, num_polygon] bool array.
    """
    # first convert polygon to directed lines
    num_points_of_polygon = polygon.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon.shape[0]
    if clockwise:
        vec1 = (
            polygon
            - polygon[
                :,
                [num_points_of_polygon - 1] + list(range(num_points_of_polygon - 1)),
                :,
            ]
        )
    else:
        vec1 = (
            polygon[
                :,
                [num_points_of_polygon - 1] + list(range(num_points_of_polygon - 1)),
                :,
            ]
            - polygon
        )
    # vec1: [num_polygon, num_points_of_polygon, 2]
    ret = np.zeros((num_points, num_polygons), dtype=np.bool_)
    success = True
    cross = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            success = True
            for k in range(num_points_of_polygon):
                cross = vec1[j, k, 1] * (polygon[j, k, 0] - points[i, 0])
                cross -= vec1[j, k, 0] * (polygon[j, k, 1] - points[i, 1])
                if cross >= 0:
                    success = False
                    break
            ret[i, j] = success
    return ret


def in_range_3d(pcds, point_range):
    """Check whether the points are in the given range.
    Args:
        point_range (list | torch.Tensor): The range of point
            (x_min, y_min, z_min, x_max, y_max, z_max)
    Note:
        In the original implementation of SECOND, checking whether
        a box in the range checks whether the points are in a convex
        polygon, we try to reduce the burden for simpler cases.
    Returns:
        torch.Tensor: A binary vector indicating whether each point is
            inside the reference range.
    """
    in_range_flags = ((pcds[:, 0] > point_range[0][0])
                    & (pcds[:, 1] > point_range[1][0])
                    & (pcds[:, 2] > point_range[2][0])
                    & (pcds[:, 0] < point_range[0][1])
                    & (pcds[:, 1] < point_range[1][1])
                    & (pcds[:, 2] < point_range[2][1]))
    return in_range_flags


def in_range_bev(boxs, box_range):
    """Check whether the boxes are in the given range.
    Args:
        box_range (list | torch.Tensor): the range of box
            (x_min, y_min, x_max, y_max)
    Note:
        The original implementation of SECOND checks whether boxes in
        a range by checking whether the points are in a convex
        polygon, we reduce the burden for simpler cases.
    Returns:
        torch.Tensor: Whether each box is inside the reference range.
    """
    in_range_flags = ((boxs[:, 0] > box_range[0][0])
                    & (boxs[:, 1] > box_range[1][0])
                    & (boxs[:, 0] < box_range[0][1])
                    & (boxs[:, 1] < box_range[1][1]))
    return in_range_flags

def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor | np.ndarray): The value to be converted.
        offset (float, optional): Offset to set the value range.
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        (torch.Tensor | np.ndarray): Value in the range of
            [-offset * period, (1-offset) * period]
    """
    limited_val = val - torch.floor(val / period + offset) * period
    return limited_val

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners

def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners

def minmax_to_corner_2d(minmax_box):
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center
    return center_to_corner_box2d(center, dims, origin=0.0)

class ObjFilterRange:
    def __init__(self, limit_range):
        self.range = limit_range

    def __call__(self, pcds, gt_boxes=None):
        if gt_boxes is None:
            return pcds, gt_boxes

        gt_boxes_bev = center_to_corner_box2d(gt_boxes[:, [0,1]], gt_boxes[:, [3,4]], gt_boxes[:, 6])
        bounding_box = minmax_to_corner_2d(np.asarray(self.range)[np.newaxis, ...])

        ret = points_in_convex_polygon_jit(gt_boxes_bev.reshape(-1, 2), bounding_box)
        mask = np.any(ret.reshape(-1, 4), axis=1)

        gt_boxes = gt_boxes[mask]
        return pcds, gt_boxes

class RandomFlip:
    def __init__(self, probability=0.5):
        self.prob = probability
    
    def __call__(self, pcds, gt_boxes=None):
        """Inputs:
            pcds: (N, C) N demotes the number of point clouds; C contains (x, y, z, i, ...)
           Output:
            pcds: (N, C)
        """
        # x, y, z, l, w, h, yaw, catid, vx, vy
        # x flip 
        if random.random() < self.prob:
            pcds[:, 1] = -pcds[:, 1]
            if gt_boxes is not None:
                gt_boxes[:, 1] = -gt_boxes[:, 1]
                gt_boxes[:, 6] = -gt_boxes[:, 6] + 2*np.pi
                if gt_boxes.shape[1] > 7:  # y axis: x, y, z, w, h, l, vx, vy, r
                    gt_boxes[:, 9] = -gt_boxes[:, 9]
    
        # y flip 
        if random.random() < self.prob:
            pcds[:, 0] = -pcds[:, 0]
            if gt_boxes is not None:
                gt_boxes[:, 0] = -gt_boxes[:, 0]
                gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi  # TODO: CHECK THIS 
        
                if gt_boxes.shape[1] > 7:  # y axis: x, y, z, w, h, l, vx, vy, r
                    gt_boxes[:, 8] = -gt_boxes[:, 8]
    
        return pcds, gt_boxes
    

class GlobalRotation:
    def __init__(self, rot_rad=np.pi):
        self.rotation = rot_rad

    def __call__(self, pcds, gt_boxes=None):
        # x, y, z, l, w, h, yaw, catid, vx, vy
        angle = np.random.uniform(-self.rotation, self.rotation)
        rot_sin = math.sin(angle)
        rot_cos = math.cos(angle)

        # rotation at Z axis
        rotz_matrix = np.asarray([[rot_cos, -rot_sin, 0.0],
                                  [rot_sin, rot_cos, 0.0],
                                  [0.0, 0.0, 1.0]], dtype=pcds.dtype)
        pcds[:, :3] = pcds[:, :3] @ rotz_matrix
        if gt_boxes is not None:
            gt_boxes[:, :3] = gt_boxes[:, :3] @ rotz_matrix
            if gt_boxes.shape[1] > 7:  # y axis: x, y, z, w, h, l, vx, vy, r
                gt_vxy = np.hstack([gt_boxes[:, 8:10], np.zeros((gt_boxes.shape[0], 1))])
                gt_vxy = gt_vxy @ rotz_matrix
                gt_boxes[:, 8:10] = gt_vxy[:, :2]
            gt_boxes[:, 6] -= angle + 2*np.pi
        return pcds, gt_boxes
    

class GlobalScale:
    def __init__(self, min_scale=0.95, max_scale=1.05):
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def __call__(self, pcds, gt_boxes=None):
        # x, y, z, l, w, h, yaw, catid, vx, vy
        noise_scale = np.random.uniform(self.min_scale, self.max_scale)
        pcds[:, :3] *= noise_scale
        if gt_boxes is not None:
            # gt_boxes[:, :6] *= noise_scale
            gt_boxes[:, 8:10] *= noise_scale
        return pcds, gt_boxes
    
class GlobalShift:
    def __init__(self, shift_range=((-3, 3), (-3, 3), (-0.4, 0.4))):
        self.shift_range = shift_range

    def __call__(self, pcds, gt_boxes=None):
        noise_translate = np.array(
            [
                np.random.uniform(self.shift_range[0][0], self.shift_range[0][1]),
                np.random.uniform(self.shift_range[1][0], self.shift_range[1][1]),
                np.random.uniform(self.shift_range[2][0], self.shift_range[2][1]),
            ]
        ).T
        pcds[:, :3] += noise_translate
        if gt_boxes is not None:
            gt_boxes[:, :3] += noise_translate
        return pcds, gt_boxes
    
class GlobalNoise:
    def __init__(self, noise_mean=0.0, noise_std=0.0):
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def __call__(self, pcds, gt_boxes=None):
        xyz_noise = np.random.normal(self.noise_mean, self.noise_std, size=(pcds.shape[0], 3))
        pcds[:, :3] = pcds[:, :3] + xyz_noise
        return pcds, gt_boxes
    
class PointsRangeFilter:
    def __init__(self,
                 range_x = (-100.0, 100.0), 
                 range_y = (-100.0, 100.0),
                 range_z = (-5.0, 3.0)):
        self.range = [range_x, range_y, range_z]

    def __call__(self, pcds, gt_boxes=None):
        pcd_mask = in_range_3d(pcds, self.range)
        pcds = pcds[pcd_mask, :]
        return pcds, gt_boxes
    
class ObjectRangeFilter:
    def __init__(self,
                 range_x = (-100.0, 100.0), 
                 range_y = (-100.0, 100.0),
                 range_z = (-5.0, 3.0)):
        self.bev_range = [range_x, range_y]

    def __call__(self, gt_boxes, gt_labels):
        mask = in_range_3d(gt_boxes, self.bev_range)
        gt_boxes = gt_boxes[mask, :]
        gt_labels = gt_labels[mask.numpy().astype(np.bool)]
        gt_boxes[:, 6] = limit_period(gt_boxes[:, 6], offset=0.5, period=2 * np.pi)

        return gt_boxes, gt_labels
    
class PointAugCompose(object):
    def __init__(self, transforms):
        self.transforms = []
        for transform in transforms:
            self.transforms.append(transform)
    def __call__(self, pcds, gt_boxes=None):
        for t in self.transforms:
            pcds, gt_boxes = t(pcds, gt_boxes)
        return pcds, gt_boxes
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = []
        for transform in transforms:
            self.transforms.append(transform)

    def __call__(self, res, info):
        for t in self.transforms:
            res, info = t(res, info)
            if res is None:
                return None
        return res, info
    

class PolarMix(object):
    def __init__(self, 
                 instance_class: List[int], 
                 swap_ratio: float = 0.5,
                 rotate_paste_ratio: float = 1.0,
                 omega_factor: float = 0.667):
        self.instance_class = instance_class
        self.swap_ratio = swap_ratio
        self.rotate_paste_ratio = rotate_paste_ratio
        self.omega_factor = omega_factor

    def swap(self, pt1, pt2, start_angle, end_angle, label1, label2):
        # calculate horizontal angle for each point
        yaw1 = -np.arctan2(pt1[:, 1], pt1[:, 0])
        yaw2 = -np.arctan2(pt2[:, 1], pt2[:, 0])

        # select points in sector
        idx1 = np.where((yaw1>start_angle) & (yaw1<end_angle))
        idx2 = np.where((yaw2>start_angle) & (yaw2<end_angle))

        # swap
        pt1_out = np.delete(pt1, idx1, axis=0)
        pt1_out = np.concatenate((pt1_out, pt2[idx2]))
        pt2_out = np.delete(pt2, idx2, axis=0)
        pt2_out = np.concatenate((pt2_out, pt1[idx1]))

        label1_out = np.delete(label1, idx1)
        label1_out = np.concatenate((label1_out, label2[idx2]))
        label2_out = np.delete(label2, idx2)
        label2_out = np.concatenate((label2_out, label1[idx1]))
        assert pt1_out.shape[0] == label1_out.shape[0]
        assert pt2_out.shape[0] == label2_out.shape[0]
        return pt1_out, label1_out

    def rotate_copy(self, pts, labels):
        # extract instance points
        pts_inst, labels_inst = [], []
        for s_class in self.instance_class:
            pt_idx = np.where((labels == s_class))
            pts_inst.append(pts[pt_idx])
            labels_inst.append(labels[pt_idx])
        pts_inst = np.concatenate(pts_inst, axis=0)
        labels_inst = np.concatenate(labels_inst, axis=0)

        # rotate-copy
        pts_copy = [pts_inst]
        labels_copy = [labels_inst]

        angle_list = [
                np.random.random() * np.pi * self.omega_factor,
                (np.random.random() + 1) * np.pi * self.omega_factor
        ]
        for omega_j in angle_list:
            rot_mat = np.array([[np.cos(omega_j),
                                 np.sin(omega_j), 0],
                                [-np.sin(omega_j),
                                 np.cos(omega_j), 0], [0, 0, 1]])
            new_pt = np.zeros_like(pts_inst)
            new_pt[:, :3] = np.dot(pts_inst[:, :3], rot_mat)
            new_pt[:, 3:] = pts_inst[:, 3:]
            pts_copy.append(new_pt)
            labels_copy.append(labels_inst)
        pts_copy = np.concatenate(pts_copy, axis=0)
        labels_copy = np.concatenate(labels_copy, axis=0)
        return pts_copy, labels_copy

    def __call__(self, pcds1, labels1, pcds2, labels2):
        pts_out, labels_out = pcds1, labels1
        # swapping
        if np.random.random() < self.swap_ratio:
            alpha = (np.random.random() - 1) * np.pi
            beta = alpha + np.pi
            pts_out,labels_out = self.swap(pcds1, pcds2, start_angle=alpha, end_angle=beta, label1=labels1, label2=labels2)

        # rotate-pasting
        if np.random.random() < self.rotate_paste_ratio:
            # rotate-copy
            pts_copy, labels_copy = self.rotate_copy(pcds2, labels2)
            # paste
            pts_out = np.concatenate((pts_out, pts_copy), axis=0)
            labels_out = np.concatenate((labels_out, labels_copy), axis=0)

        return pts_out, labels_out    