import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import optimizer, lr_scheduler
from typing import Any, Dict, Iterable, List, Optional, Union
from torch.utils.data.distributed import DistributedSampler


import numpy as np
import collections
import logging
import random
import yaml
import sys

from tqdm import tqdm
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from utils.logger import config_logger
import gc 

import pdb

@torch.no_grad()
def reduce_tensor(inp, group=None):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = torch.distributed.get_world_size(group)
    if world_size < 2:
        return inp
    
    if isinstance(inp, torch.Tensor):
        reduce_inp = inp.cpu()
    else:
        reduce_inp = inp
    
    object_list = [None for i in range(world_size)]
    torch.distributed.all_gather_object(object_list, reduce_inp, group=group)
    return sum(object_list) / world_size

def dic_to_human_readable_str(title_name, param_dic):
    assert isinstance(param_dic, dict)
    string = "{}:\n".format(title_name)
    for key in param_dic:
        string += "{0}:\t{1}\n".format(key, param_dic[key])
    return string

class ADDistTrainer(object):
    def __init__(self, detect_anomaly=False,
                sync_batchnorm=True,
                clip_grad_norm=None,
                clip_grad_norm_type=None,
                precision=16):
        # capture arguments to provide to context
        assert precision in [16, 32], "'precision' must be 16 or 32, but got {}".format(precision)
        self.enable_fp16 = (precision == 16)
        self.detect_anomaly = detect_anomaly
        self.sync_batchnorm = sync_batchnorm
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_norm_type = clip_grad_norm_type
        self.local_rank = int(os.getenv("LOCAL_RANK"))
        self.global_rank = 0

        self.build_log()
        self.build_env()
        self.set_randoom_seed()
        self.build_training_prop()

        # save settings
        # self.log_str(dic_to_human_readable_str("Project Settings", kwargs))

    def build_log(self):
        raise NotImplementedError

    def build_env(self):
        # set env
        try:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.process_group = torch.distributed.group.WORLD
            self.world_size = torch.distributed.get_world_size(self.process_group)
            self.global_rank = torch.distributed.get_rank(self.process_group)
        except:
            self.global_rank = 0
            self.world_size =1
        self.device = torch.device('cuda:{}'.format(self.local_rank))
        torch.cuda.set_device(self.local_rank)

    def set_randoom_seed(self):
        # reset random seed
        seed = self.global_rank*4 + 50051
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def build_training_prop(self):
        # set training properties
        self.base_net: nn.Module = None
        self.model: nn.Module = None
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.test_dataloader: Optional[DataLoader] = None
        self.optimizer: Union[optimizer.Optimizer, Dict[str, optimizer.Optimizer]] = None
        self.scheduler: Union[lr_scheduler._LRScheduler, Dict[str, lr_scheduler._LRScheduler]] = None
        self.global_step_idx: int = 0
        self.step_idx: int = 0
        self.epoch_idx: int = 0
        self.max_epochs: int = 0
        self.best_ckpt: Optional[str]= None
        self.best_value: float = 0.0
        self.best_epoch: int = 0
        self.disable_prefix: list = []
        self.limit_train_iters: int = -1
        self.limit_val_iters: int = -1
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_fp16)
    
    def build_dataset(self):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))

    def build_network(self):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))

    def build_model(self):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))

    def build_metric(self):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))

    def build_dataloader(self):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))

    def set_attrs(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
    
    def save_hyperparameters(self, param_dic: Dict[str, Any], fn_yaml: str):
        if (self.global_rank == 0) and (self.recorder is not None):
            assert isinstance(param_dic, dict)
            fname_yaml = os.path.join(self.recorder.log_dir, fn_yaml)
            with open(fname_yaml, 'w') as f:
                f.write(yaml.dump(param_dic, allow_unicode=True))
    
    def log(self, enable_sync_dist: bool = True, ignore_log_step: bool = False, **kwargs):
        if (((self.step_idx % self.log_every_n_steps) == 0) or ignore_log_step) and (self.recorder is not None):
            if enable_sync_dist:
                kwargs_sync = collections.OrderedDict()
                for key in kwargs:
                    kwargs_sync[key] = reduce_tensor(kwargs[key], self.process_group)
                
                self.recorder.record(kwargs_sync, self.epoch_idx, self.step_idx, self.global_step_idx)
            else:
                self.recorder.record(kwargs, self.epoch_idx, self.step_idx, self.global_step_idx)
    
    def log_str(self, string: str):
        if self.recorder is not None:
            self.recorder.record_str(string)
    
    def configure_optimizers(self):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def training_step(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def validation_step(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def test_step(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def predict_step(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def training_epoch_end(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def validation_epoch_end(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def test_epoch_end(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def predict_epoch_end(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def training_end(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))

    def load_from_checkpoint(self,
                            fname_ckpt: str,
                            strict:bool = True):
        saved_dic = torch.load(fname_ckpt, map_location='cpu')
        self.base_net.load_state_dict(saved_dic['model_dic'], strict=strict)
        self.logger.info("Load ckpt from {}".format(fname_ckpt))

    def save_checkpoint(self, fname_ckpt: str):
        model_dic = None
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_dic = self.model.module.state_dict()
        else:
            model_dic = self.model.state_dict()
        
        saved_dic = {
            'model_dic': model_dic,
            'global_step_idx': self.global_step_idx,
            'step_idx': self.step_idx,
            'epoch_idx': self.epoch_idx
        }
        if self.optimizer is not None:
            saved_dic.update({
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            })
        torch.save(saved_dic, fname_ckpt)
        self.logger.info("save model to {}".format(fname_ckpt))
    
    def resume_from_checkpoint(self, fname_ckpt: str):
        self.logger.info("resume model from {}".format(fname_ckpt))
        saved_dic = torch.load(fname_ckpt, map_location='cpu')
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(saved_dic['model_dic'], strict=True)
        else:
            self.model.load_state_dict(saved_dic['model_dic'], strict=True)

        try:
            opt_dict = saved_dic['optimizer'].state_dict()
            sch_dict = saved_dic['scheduler'].state_dict()
        except:
            opt_dict = saved_dic['optimizer']
            sch_dict = saved_dic['scheduler']
        # sch_dict['total_steps'] = int(sch_dict['total_steps']*1.25)

        self.optimizer.load_state_dict(opt_dict)
        self.scheduler.load_state_dict(sch_dict)
        self.global_step_idx = saved_dic['global_step_idx']+1
        self.step_idx = saved_dic['step_idx']
        self.epoch_idx = saved_dic['epoch_idx']+1
    
    def sync_all_process(self):
        torch.distributed.barrier()
    
    def fit(self, ckpt_path: str = None):
        # configure optimizer and scheduler
        self.configure_optimizers()

        self.logger.info("rank: {}/{}; batch_size: {}".format(self.global_rank, self.world_size, self.pGen.batch_size_per_gpu))
        # resume from checkpoint
        if ckpt_path is not None:
            self.resume_from_checkpoint(ckpt_path)
        
        # logger model, optimizer, scheduler
        if self.global_rank == 0:
            self.logger.info(self.model)
            self.logger.info(self.optimizer)
            self.logger.info(self.scheduler)

        # judge optimizer and scheduler
        if not isinstance(self.optimizer, optimizer.Optimizer):
            assert isinstance(self.optimizer, dict) and \
            all([isinstance(self.optimizer[key], optimizer.Optimizer) for key in self.optimizer]), \
            "self.optimizer must be torch.optim.optimizer.Optimizer or dict[str, torch.optim.optimizer.Optimizer]"
        
        if not isinstance(self.scheduler, lr_scheduler._LRScheduler):
            assert isinstance(self.scheduler, dict) and \
            all([isinstance(self.scheduler[key], lr_scheduler._LRScheduler) for key in self.scheduler]), \
            "self.scheduler must be torch.optim.lr_scheduler._LRScheduler or dict[str, torch.optim.lr_scheduler._LRScheduler]"
        
        # start fitting
        print('========fp16:{}========='.format(self.enable_fp16))
        while(self.epoch_idx < self.max_epochs):
            # training
            self.model.train()
            self.train_sampler.set_epoch(self.epoch_idx)
            is_aug = True
            if self.epoch_idx >= self.aug_epochs:
                is_aug = False
            self.train_dataset.reset_sample(is_aug=is_aug)
            self.step_idx = 0
            with torch.autograd.set_detect_anomaly(self.detect_anomaly):
                loop = enumerate(self.train_dataloader)
                if self.global_rank == 0:
                    self.logger.info('Training Epoch:{}'.format(self.epoch_idx))
                    loop = tqdm(loop, desc="Training Epoch: {}".format(self.epoch_idx), total=len(self.train_dataloader), ascii=True)
                
                for batch_idx, batch in loop:
                    if ((self.limit_train_iters>0) and (batch_idx >= self.limit_train_iters)):
                        break

                    self.optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=self.enable_fp16, cache_enabled=self.enable_fp16):
                        loss = self.training_step(batch_idx, batch)
                    
                    if loss is not None:
                        assert (isinstance(loss, torch.Tensor) and loss.numel() == 1)
                        # automatic backward
                        # loss = torch.nan_to_num(loss, nan=500.0, posinf=1000.0, neginf=-1000.0)
                        self.scaler.scale(loss).backward()
                        if self.clip_grad_norm is not None:
                            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm, norm_type=self.clip_grad_norm_type)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                    
                    self.global_step_idx += 1
                    self.step_idx += 1
            
            # self.sync_all_process()
            self.training_epoch_end()
            torch.cuda.empty_cache()
            gc.collect()

            # evaluate
            if self.val_dataloader is not None:
                if (self.epoch_idx+1) % self.check_val_every_n_epoch == 0:
                    self.model.eval()
                    loop = enumerate(self.val_dataloader)
                    if self.global_rank == 0:
                        self.logger.info("Evaluating Epoch: {}".format(self.epoch_idx))
                        loop = tqdm(loop, desc="Evaluating Epoch: {}".format(self.epoch_idx), total=len(self.val_dataloader), ascii=True)
                    
                    for batch_idx, batch in loop:
                        if ((self.limit_val_iters>0) and (batch_idx >= self.limit_val_iters)):
                            break

                        with torch.cuda.amp.autocast(enabled=self.enable_fp16, cache_enabled=self.enable_fp16):
                            self.validation_step(batch_idx, batch)
                    # self.sync_all_process()
                    self.validation_epoch_end()
            else:
                pass
            
            # self.sync_all_process()
            torch.cuda.empty_cache()
            gc.collect()
            self.epoch_idx += 1

        self.training_end()

    @torch.no_grad()
    def data_test(self):
        self.model.eval()
        loop = enumerate(self.train_dataloader)
        if self.global_rank == 0:
            self.logger.info('Training Epoch:{}'.format(self.epoch_idx))
            loop = tqdm(loop, desc="Training Epoch: {}".format(self.epoch_idx), total=len(self.train_dataloader), ascii=True)
        for batch_idx, batch in loop:
            if ((self.limit_train_iters>0) and (batch_idx >= self.limit_train_iters)):
                break
    
            with torch.cuda.amp.autocast(enabled=self.enable_fp16, cache_enabled=self.enable_fp16):
                self.training_step(batch_idx, batch)

        # self.sync_all_process()
        self.training_epoch_end()

        if self.val_dataloader is not None:
            loop = enumerate(self.val_dataloader)
            if self.global_rank == 0:
                self.logger.info("Evaluating Epoch: {}".format(self.epoch_idx))
                loop = tqdm(loop, desc="Evaluating Epoch: {}".format(self.epoch_idx), total=len(self.val_dataloader), ascii=True)
                    
            for batch_idx, batch in loop:
                if ((self.limit_val_iters>0) and (batch_idx >= self.limit_val_iters)):
                    break
                with torch.cuda.amp.autocast(enabled=self.enable_fp16, cache_enabled=self.enable_fp16):
                    self.validation_step(batch_idx, batch)
            # self.sync_all_process()
            self.validation_epoch_end()

        self.training_end()

    @torch.no_grad()
    def validate(self, ckpt_path: str = None):        
        # resume from checkpoint
        if ckpt_path is not None:
            self.resume_from_checkpoint(ckpt_path)
        
        # evaluate
        self.model.eval()
        loop = enumerate(self.val_dataloader)
        if self.global_rank == 0:
            loop = tqdm(loop, desc="Evaluating Epoch: {}".format(self.epoch_idx), total=len(self.val_dataloader), ascii=True)
        
        for batch_idx, batch in loop:
            with torch.cuda.amp.autocast(enabled=self.enable_fp16, cache_enabled=self.enable_fp16):
                self.validation_step(batch_idx, batch)
        # self.sync_all_process()
        self.validation_epoch_end()
    
    @torch.no_grad()
    def test(self, ckpt_path: str = None):        
        # resume from checkpoint
        if ckpt_path is not None:
            self.resume_from_checkpoint(ckpt_path)

        # evaluate
        self.model.eval()
        loop = enumerate(self.val_dataloader)
        if self.global_rank == 0:
            loop = tqdm(loop, desc="Testing Epoch: {}".format(self.epoch_idx), total=len(self.val_dataloader), ascii=True)
        
        for batch_idx, batch in loop:
            with torch.cuda.amp.autocast(enabled=self.enable_fp16, cache_enabled=self.enable_fp16):
                self.test_step(batch_idx, batch)
        # self.sync_all_process()
        self.test_epoch_end()
