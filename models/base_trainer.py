import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy
import yaml
import time
import sys
import os

import collections

from .model_utils import builder
from utils.config_parser import get_module, class2dic, class2dic_iterative

from runners.trainer import ADDistTrainer
import shutil
from utils.logger import config_logger
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import datasets
import models

import pickle

import pdb

@torch.no_grad()
def gather_metric(metric_func, group=None):
    world_size = torch.distributed.get_world_size(group)
    if world_size < 2:
        return metric_func
    for name, tensor in metric_func.all_tensors.items():
        tensor = torch.from_numpy(tensor).cuda()
        with torch.no_grad():
            tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(tensor_list, tensor, group=group)
            # tensor = torch.cat(tensor_list, 0).cpu().numpy()
            tensor = torch.sum(torch.stack(tensor_list), dim=0).cpu().numpy()
            metric_func.all_tensors[name] = tensor
    return metric_func

@torch.no_grad()
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    torch.distributed.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

class BaseTrainer(ADDistTrainer):
    def __init__(self, args=None, config=None, 
                detect_anomaly: bool = False,
                sync_batchnorm: bool = True,
                clip_grad_norm: float = None,
                clip_grad_norm_type: int = None,
                is_validate: bool = False):
        self.args = args
        self.config = config
        self.precision = args.precision
        self.pGen, self.pDataset, self.pModel = self.config.get_config()
        super(BaseTrainer, self).__init__(detect_anomaly, 
                                          sync_batchnorm, 
                                          clip_grad_norm, 
                                          clip_grad_norm_type, 
                                          args.precision)

        # save model config
        # save_dict = class2dic_iterative(self.pModel)
        # self.save_hyperparameters(save_dict, "hparams.yaml")
        # self.log(enable_sync_dist=False, ignore_log_step=True, **kwargs)

        self.max_point_num = self.pGen.max_point_num
        self.num_workers = self.pDataset.num_workers
        self.epoch_idx = self.pModel.scheduler.begin_epoch
        self.max_epochs = self.pModel.scheduler.max_epochs
        self.disable_prefix = self.pModel.disable_list
        self.check_val_every_n_epoch = self.pModel.val_epochs
        if hasattr(self.pModel, 'aug_epochs'):
            self.aug_epochs = self.pModel.aug_epochs
        else:
            self.aug_epochs = 9999
        self.monitor_keys = self.pModel.monitor_keys
        self.is_qat = self.args.is_qat
        self.dla_core = self.args.dla_core
        self.pretrain_model = self.args.pretrain_model
        self.is_validate = is_validate
        self.sync_batchnorm = sync_batchnorm
        self.build_dataset()
        self.build_network()
        self.build_model()
        self.build_dataloader()
        self.build_loss()
        self.build_metric()
    
    def build_log(self):
        prefix = self.pGen.name
        time_version = self.pGen.time_version
        self.save_path = os.path.join("experiments", time_version, prefix, self.args.version)
        self.model_prefix = os.path.join(self.save_path, "checkpoint")
        os.system('mkdir -p {}'.format(self.model_prefix))
        self.best_path=self.model_prefix
        if self.local_rank == 0:
            cfg_path = os.path.join(self.save_path, '{}.py'.format(prefix))
            try:
                shutil.copyfile(self.args.config, cfg_path)
            except shutil.SameFileError:
                pass
            self.writer = SummaryWriter(self.save_path)
            self.fpath = os.path.join(self.save_path, "metric.txt")
            fp = open(self.fpath, 'w')
            fp.close()
        # start logging
        config_logger(os.path.join(self.save_path, "log.txt"))
        self.logger = logging.getLogger()


    def build_dataloader(self):
        if not self.is_validate:
            try:
                self.train_sampler = DistributedSampler(self.train_dataset)
            except:
                self.train_sampler = None
            self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.pGen.batch_size_per_gpu,
                                           shuffle=(self.train_sampler is None),
                                           num_workers=self.num_workers,
                                           sampler=self.train_sampler,
                                           pin_memory=True)

            self.train_iters = len(self.train_dataloader)
            self.limit_train_iters = self.args.limit_train_batches

        # define dataloader
        try:
            self.val_sampler = DistributedSampler(self.val_dataset)
        except:
            self.val_sampler = None
        self.val_dataloader = DataLoader(self.val_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     sampler=self.val_sampler,
                                     pin_memory=True)
        self.val_iters = len(self.val_dataloader)
        self.limit_val_iters = self.args.limit_val_batches
            
    def build_dataset(self):
        # define dataloader
        if not self.is_validate:
            self.train_dataset = eval(self.pDataset.Train.type)(self.pDataset.Train)
        self.val_dataset = eval(self.pDataset.Val.type)(self.pDataset.Val)
        self.val_dataset.reset_sample()

    def build_network(self):
        self.base_net = eval(self.pModel.model_type)(pModel=self.pModel)
        pretrain_model = self.args.pretrain_model
        if pretrain_model is not None:
            model_dic = torch.load(pretrain_model, map_location='cpu')['model_dic']
            self.base_net.load_state_dict(model_dic, strict=False)
            self.logger.info("Load model from {}".format(pretrain_model))

    def build_model(self):
        assert isinstance(self.base_net, nn.Module), "'model' must belong to nn.Module, but got {}".format(type(self.base_net))
        model = self.base_net
        try:
            if self.sync_batchnorm:
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model.to(self.device),
                                                    device_ids=[self.local_rank],
                                                    output_device=self.local_rank,
                                                    broadcast_buffers=False,
                                                    find_unused_parameters=True)
        except:
            self.model = model

    def build_loss(self):
        self.loss_funcs = {}
        for (key, loss_cfg) in class2dic(self.pModel.LossParam).items():
            self.loss_funcs.update({key: get_module(loss_cfg)})
    
    def build_metric(self):
        self.metric_dics = {}
        for (key, metric_cfg) in class2dic(self.pModel.MetricParam).items():
            self.metric_dics.update({key: get_module(metric_cfg)})

    def training_epoch_start(self):
        pass
    
    def training_step(self):
        raise NotImplementedError
    
    def validation_step(self):
        raise NotImplementedError
    
    def validation_epoch_end(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        self.optimizer = builder.get_optimizer(self.pModel.optimizer, self.model)
        self.scheduler = builder.get_scheduler(self.optimizer, self.pModel.scheduler, self.train_iters)
    
    def deploy_trt(self):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def fill_tensor(self, pt_tensor, fill_params=None, fill_value=0):
        '''
        pt_tensor: torch.Tensor
        fill_params: dict, axis: max_length
        '''
        assert isinstance(pt_tensor, torch.Tensor), "'pt_tensor' must be torch.Tensor, but got {}".format(type(pt_tensor))
        if fill_params is None:
            return pt_tensor
        else:
            padding_params = []
            for i in range(pt_tensor.dim()):
                if i not in fill_params:
                    padding_params.append((0, 0))
                else:
                    padding_length = fill_params[i] - pt_tensor.shape[i]
                    assert padding_length >= 0,\
                    "The padding length: {0} of axis: {1} must be no less than {2}".format(fill_params[i], i, pt_tensor.shape[i])
                    padding_params.append((0, padding_length))
            
            pt_tensor_np = np.pad(pt_tensor.detach().cpu().numpy(), tuple(padding_params), 'constant', constant_values=fill_value)
            return torch.from_numpy(pt_tensor_np).to(pt_tensor)
    
    def test_time(self, model, *args, **kwargs):
        time_cost = []
        with torch.no_grad():
            for i in range(100):
                start = time.time()
                torch.cuda.synchronize()
                model(*args, **kwargs)
                torch.cuda.synchronize()
                end = time.time()
                time_cost.append(end - start)
        
        tc = float(np.array(time_cost[20:]).mean())
        return tc