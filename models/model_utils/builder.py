from torch.optim import SGD, AdamW, lr_scheduler

from functools import partial
import math

import pdb


def schedule_with_warmup(k, num_epoch, per_epoch_num_iters, pct_start, step, decay_factor):
    warmup_iters = int(num_epoch * per_epoch_num_iters * pct_start)
    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        epoch = k // per_epoch_num_iters
        step_idx = (epoch // step)
        return math.pow(decay_factor, step_idx)


def get_scheduler(optimizer, pSch, per_epoch_num_iters):
    num_epoch = pSch.max_epochs
    if pSch.type == 'OneCycle':
        base_lr = optimizer.state_dict()['param_groups'][0]['lr']
        max_lrs = []
        for i, group in enumerate(optimizer.param_groups):
            max_lrs.append(group['lr'])
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=max_lrs,
                                            epochs=num_epoch, steps_per_epoch=per_epoch_num_iters,
                                            pct_start=pSch.pct_start, anneal_strategy='cos',
                                            div_factor=10, final_div_factor=1e4)
        return scheduler
    elif pSch.type == 'step':
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(
                schedule_with_warmup,
                num_epoch=num_epoch,
                per_epoch_num_iters=per_epoch_num_iters,
                pct_start=pSch.pct_start,
                step=pSch.step,
                decay_factor=pSch.decay_factor
            ))
        return scheduler
    else:
        raise NotImplementedError(pSch.type)


def get_optimizer(pOpt, model):
    seg_params, det_params, base_params, img_params = [], [], [], []
    for name, param in model.named_parameters():
        if 'det_head' in name:
            det_params.append(param)
        elif 'seghead' in name:
            seg_params.append(param)
        elif 'image_net' in name:
            img_params.append(param)
        else:
            base_params.append(param)

    seg_lr = pOpt.base_lr
    if hasattr(pOpt, 'seg_lr'):
        seg_lr = pOpt.seg_lr

    det_lr = pOpt.base_lr
    if hasattr(pOpt, 'det_lr'):
        det_lr = pOpt.det_lr

    img_lr = pOpt.base_lr
    if hasattr(pOpt, 'img_lr'):
        img_lr = pOpt.img_lr

    opt_params = [{'params': base_params},
                  {'params': seg_params, 'lr': seg_lr},
                  {'params': img_params, 'lr': img_lr},
                  {'params': det_params, 'lr': det_lr}]

    if pOpt.type in ['adam', 'adamw']:
        optimizer = AdamW(params=opt_params,
                        lr=pOpt.base_lr,
                        weight_decay=pOpt.wd)
        return optimizer
    elif pOpt.type == 'sgd':
        optimizer = SGD(params=opt_params,
                        lr=pOpt.base_lr,
                        momentum=pOpt.momentum,
                        weight_decay=pOpt.wd,
                        nesterov=pOpt.nesterov)
        return optimizer

    else:
        raise NotImplementedError(pOpt.type)