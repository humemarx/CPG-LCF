import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import datasets
from torch.utils.data import DataLoader
from models.lidar_seg.fusion_seg_trainer import FusionSegTrainer
from models.lidar_seg.fusion_e2e_trainer import FusionE2ETrainer

import argparse
import importlib

import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False

import pdb

def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel = config.get_config()
    model_trainer = eval(pModel.trainer_type)(args=args, config=config, sync_batchnorm=True, detect_anomaly=False, is_validate=args.is_validate)
    if args.is_validate:
        if args.is_test:
            model_trainer.test(ckpt_path=args.resume_ckpt)
        else:
            model_trainer.validate(ckpt_path=args.resume_ckpt)
    else:
        model_trainer.fit(ckpt_path=args.resume_ckpt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--version', help='version name of the experiment', type=str, default=None)
    parser.add_argument('--is_qat', help='version name of the experiment', action='store_true')
    parser.add_argument('--precision', help='precision of model training', type=int, default=16)
    parser.add_argument('--dla_core', help='the core index of DLA', type=int, default=None)

    parser.add_argument('--pretrain_model', help='pretrain model', type=str, default=None)
    parser.add_argument('--resume_ckpt', help='resume checkpoint', type=str, default=None)
    parser.add_argument('--limit_train_batches', help='train batches', type=int, default=0)
    parser.add_argument('--limit_val_batches', help='val batches', type=int, default=0)
    parser.add_argument('--is_validate', help='is validate', action='store_true')
    parser.add_argument('--is_test', help='is validate', action='store_true')

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)