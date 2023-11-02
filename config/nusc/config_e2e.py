import os
import numpy as np
import torch.nn as nn

def get_config():
    class General:
        log_frequency = 100
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        time_version = '07-25-14'

        batch_size_per_gpu = 4
        SeqDir = './data/nuscenes'
        category_list = ['ignore', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
        ignore_index = 0

        det_tasks = [
            dict(num_class=1, class_names=["car"], class_idxs=[0]),
            dict(num_class=2, class_names=["truck", "construction_vehicle"], class_idxs=[1,4]),
            dict(num_class=2, class_names=["bus", "trailer"], class_idxs=[3,2]),
            dict(num_class=1, class_names=["barrier"], class_idxs=[9]),
            dict(num_class=2, class_names=["motorcycle", "bicycle"], class_idxs=[6,5]),
            dict(num_class=2, class_names=["pedestrian", "traffic_cone"], class_idxs=[7,8]),
        ]
        cam_list = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        use_aux = False
        use_swa = False
        use_image_save = False
        use_consistency = True
        kd_method = 'none'
        test_flip = False
        test_gt = False
        max_point_num = 40000

        class ModalParam:
            lidar_num = 7
            modal_list = ('camera_raw',)
            modal_nums = (256,)

        class Voxel:
            rv_shape = (64, 2048)
            bev_shape = (512, 512, 30)
            out_size_factor = 4
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
            bev_params=dict(
                range_x = (-51.2, 51.2),
                range_y = (-51.2, 51.2),
                range_z = (-5.0, 3.0),
                size = (512, 512, 30)
            )
            rv_params=dict(
                phi_range=(-180.0, 180.0),
                theta_range = (-40.0, 20.0),
                size = (64, 2048)
            )
    
    class DatasetParam:
        num_workers = 4
        class Train:
            use_consistency = General.use_consistency
            mode = 'train'
            type = 'datasets.nusc.nusc_cam_data.DataloadTrain'
            num_workers = 4
            frame_point_num = [40000]
            SeqDir = General.SeqDir
            fname_pkl = os.path.join(SeqDir, 'nutcp_infos_train.pkl')

            use_prob_resample = False
            obj_sample = False
            det_tasks = General.det_tasks
            use_aux = General.use_aux
            Voxel = General.Voxel
            rand_level = 0
            max_objs = 500
            # filter obj from lidaseg
            filter_list = [2, 12, 13]

            class PointAug:
                transforms = [
                    # dict(type="ObjFilterRange", params=dict(limit_range=[-51.2, -51.2, 51.2, 51.2])),
                    dict(type="GlobalNoise", params=dict(noise_mean=0.0, noise_std=0.001)), 
                    dict(type="GlobalShift", params=dict(shift_range=((-3, 3), (-3, 3), (-0.4, 0.4)))), 
                    dict(type="GlobalScale", params=dict(min_scale=0.95, max_scale=1.05)), 
                    dict(type="RandomFlip", params=dict(probability=0.5)), 
                    dict(type="GlobalRotation", params=dict(rot_rad=np.pi))
                ]

            camera_norm_mean = np.asarray([123.675, 116.28, 103.53])
            camera_norm_std = np.asarray([58.395, 57.12, 57.375])
            class CameraAug:
                transforms = [
                    dict(type='RandomScale', params=dict(scale=0.712)),
                    dict(type='RandomCrop', params=dict(cropsize=(960, 640))),  
                    dict(type='HorizontalFlip', params=dict(p=0.5)),
                    # dict(type='VerticalFlip', params=dict(p=0.1)),
                    # dict(type='RotateImage', params=dict(rotate_boundary = [-5.4, 5.4])),
                    # dict(type='ColorJitter', params=dict(brightness = 0.5, contrast = 0.5, saturation = 0.5)),
                    # dict(type='GaussBlur', params=dict(radius = (0, 1, 2, 3))),
                    dict(type='ImgNormalize', params=dict(mean=np.asarray([123.675, 116.28, 103.53]), 
                                                          std=np.asarray([58.395, 57.12, 57.375])))
                ]

            class SensorParam:
                modal_list = General.ModalParam.modal_list
                camera_method = 'sum'
                camera_feat_num = 13
                camera_mask_ratio = 0.0

            class CopyPasteAug:
                is_use = True
                ObjBankDir = os.path.join(General.SeqDir, 'nusc_bank')
                paste_max_obj_num = 20
                category_list = ('barrier', 
                                 'bicycle', 
                                 'bus', 
                                 'car', 
                                 'construction_vehicle', 
                                 'motorcycle', 
                                 'pedestrian', 
                                 'traffic_cone', 
                                 'trailer', 
                                 'truck')
                road_idx = [11, 13]
                things_label_range = [1, 10]
                min_pts_num = 10

        class Val:
            mode = 'test'
            type = 'datasets.nusc.nusc_cam_data.DataloadVal'
            num_workers = 4
            frame_point_num = [None]
            SeqDir = General.SeqDir
            rand_level = 0
            fname_pkl = os.path.join(SeqDir, 'nutcp_infos_val{}.pkl'.format(rand_level))

            det_tasks = General.det_tasks
            test_flip = General.test_flip
            Voxel = General.Voxel
            max_objs = 500

            camera_norm_mean = np.asarray([123.675, 116.28, 103.53])
            camera_norm_std = np.asarray([58.395, 57.12, 57.375])
            # filter obj from lidaseg
            filter_list = [2, 12, 13]

            class CameraAug:
                transforms = [
                    dict(type='RandomScale', params=dict(scale=0.712)),
                    dict(type='CenterCrop', params=dict(cropsize=(960, 640))),  
                    dict(type='ImgNormalize', params=dict(mean=np.asarray([123.675, 116.28, 103.53]), 
                                                          std=np.asarray([58.395, 57.12, 57.375])))
                ]

            class SensorParam:
                modal_list = General.ModalParam.modal_list
                camera_method = 'sum'
                camera_feat_num = 13
                camera_mask_ratio = 0.0
    
    class ModelParam:
        model_type = "models.lidar_seg.cpgnet_e2e.CPGNet"
        trainer_type = "models.lidar_seg.fusion_e2e_trainer.FusionE2ETrainer"
        Voxel = General.Voxel
        category_list = General.category_list
        class_num = len(category_list)
        ignore_index = General.ignore_index
        point_feat_out_channels = (64, 96)
        use_aux = General.use_aux
        use_camera_raw = 'camera_raw' in General.ModalParam.modal_list
        cam_fusion = 'max'

        monitor_keys = dict(major='metric2', minor='mIoU')
        ModalParam = General.ModalParam
        disable_list = []
        with_bev = True
        with_rv = True

        class ImageBranch:
            type = 'models.segmentor2d.EncoderDecoder'
            backbone=dict(
                type='models.backbone2d.STDCContextPathNet',
                backbone_cfg=dict(
                    type='models.backbone2d.STDCNet',
                    stdc_type='STDCNet2',
                    in_channels=3,
                    channels=(32, 64, 256, 512, 1024),
                    bottleneck_type='cat',
                    num_convs=4,
                    norm_type=nn.BatchNorm2d,
                    act_type=nn.ReLU,
                    with_final_conv=False
                ),
                last_in_channels=(1024, 512),
                out_channels=128,
                ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4),
                frozen_stages = -1,
                pretrained=None,
            )
            decode_head=dict(
                type='models.seg_heads.FCNHead',
                in_channels=256,
                channels=256,
                num_convs=1,
                num_classes=19,
                in_index=3,
                concat_input=False,
                dropout_ratio=0.1,
                norm_type=nn.BatchNorm2d,
                align_corners=True,
                frozen_stages = -1,
                use_seg_label=False,
                pretrained=None,
            )
            frozen_stages = -1
            enable_fp16 = True
            pretrained = 'ckpts/stdc2_in1k-pre_512x1024_80k_cityscapes_20220224_073048-1f8f0f6c.pth'

        class BEVParam:
            type = 'models.networks.fcn_2d.FCN2D'
            base_block = 'models.networks.backbone.BasicBlock'
            fpn_block = 'models.networks.fpn.AttMerge'
            base_channels = (64, 32, 64, 128)
            base_layers = (2, 3, 4)
            base_strides = (2, 2, 2)
            scale_rate_list = [(0.5, 0.5), (0.5, 0.5)]
        
        class RVParam:
            type = 'models.networks.fcn_2d.FCN2D'
            base_block = 'models.networks.backbone.BasicBlock'
            fpn_block = 'models.networks.fpn.AttMerge'
            base_channels = (64, 32, 64, 128)
            base_layers = (2, 3, 4)
            base_strides = ((1, 2), 2, 2)
            scale_rate_list = [(1.0, 0.5), (1.0, 0.5)]
        
        class FusionParam:
            type = 'models.networks.point_fusion.PointFusion'
        
        class SpatialFusionParam:
            type = 'models.networks.point_fusion.SpatialFusion'

        class LossParam:
            class LossCE:
                type = 'models.loss.CE_OHEM'
                top_ratio = 0.2
                top_weight = 3.0
                ignore_index = General.ignore_index
                weight = 1.0
            
            class LossLS:
                type = 'models.loss.LovaszSoftmax'
                ignore_index = General.ignore_index
                weight = 3.0

            class LossConsist:
                type = 'models.loss.ConsistencyLoss'
                weight = 1.0

            class LossCamera:
                type = 'models.loss.CE_OHEM'
                top_ratio = 0.2
                top_weight = 3.0
                ignore_index = General.ignore_index
                weight = 0.5

        class MetricParam:
            class metric1:
                type = 'models.model_utils.metric.MultiClassMetric'
                ignore_index = General.ignore_index
                Classes = General.category_list
                stage = 0
                metric_key = 'mIoU'

            class metric2:
                type = 'models.model_utils.metric.MultiClassMetric'
                ignore_index = General.ignore_index
                Classes = General.category_list
                stage = 1
                metric_key = 'mIoU'

        det_epochs = 4
        val_epochs = 4
        aug_epochs = 48
        class optimizer:
            type = "adam"
            base_lr = 0.001
            img_lr = 0.001
            momentum = 0.9
            nesterov = True
            wd = 1e-3
        
        class scheduler:
            type = "OneCycle"
            begin_epoch = 0
            max_epochs = 48
            pct_start = 0.3
            final_lr = 1e-6
            step = 10
            decay_factor = 0.1

    return General, DatasetParam, ModelParam