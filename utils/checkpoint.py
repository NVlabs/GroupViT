# -------------------------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE
#
# Written by Ze Liu, Zhenda Xie
# Modified by Jiarui Xu
# -------------------------------------------------------------------------

import os
from collections import defaultdict

import torch
import torch.distributed as dist
from mmcv.runner import CheckpointLoader
from omegaconf import read_write

from .logger import get_logger

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_checkpoint(config, model, optimizer, lr_scheduler):
    logger = get_logger()
    logger.info(f'==============> Resuming form {config.checkpoint.resume}....................')
    checkpoint = CheckpointLoader.load_checkpoint(config.checkpoint.resume, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    metrics = defaultdict(float)
    if (not config.evaluate.eval_only and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint
            and 'epoch' in checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        with read_write(config):
            config.train.start_epoch = checkpoint['epoch'] + 1
        if 'amp' in checkpoint and config.train.amp_opt_level != 'O0' and checkpoint[
                'config'].train.amp_opt_level != 'O0':
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.checkpoint.resume}' (epoch {checkpoint['epoch']})")
        metrics = checkpoint['metrics']

    del checkpoint
    torch.cuda.empty_cache()
    return metrics


def save_checkpoint(config, epoch, model, metrics, optimizer, lr_scheduler, suffix=''):
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'metrics': metrics,
        'epoch': epoch,
        'config': config
    }
    logger = get_logger()

    for k, v in metrics.items():
        save_state[k] = v

    if config.train.amp_opt_level != 'O0':
        save_state['amp'] = amp.state_dict()

    if len(suffix) > 0 and not suffix.startswith('_'):
        suffix = '_' + suffix
    filename = f'ckpt_epoch_{epoch}{suffix}.pth'

    save_path = os.path.join(config.output, filename)
    logger.info(f'{save_path} saving......')
    torch.save(save_state, save_path)
    torch.save(save_state, os.path.join(config.output, 'checkpoint.pth'))
    logger.info(f'{save_path} saved !!!')

    if config.checkpoint.max_kept > 0:
        if epoch >= config.checkpoint.max_kept:
            logger.info(f'Epoch: {epoch}, greater than config.checkpoint.max_kept: {config.checkpoint.max_kept}')
            end_clean_epoch = epoch - config.checkpoint.max_kept
            old_path_list = []
            for cur_clean_epoch in range(end_clean_epoch + 1):
                old_path = os.path.join(config.output, f'ckpt_epoch_{cur_clean_epoch}{suffix}.pth')
                if os.path.exists(old_path):
                    logger.info(f'old checkpoint path {old_path} exits')
                    old_path_list.append(old_path)
            for old_path in old_path_list[:-config.checkpoint.max_kept]:
                os.remove(old_path)
                logger.info(f'old checkpoint path {old_path} removed!!!')


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    if os.path.exists(os.path.join(output_dir, 'checkpoint.pth')):
        return os.path.join(output_dir, 'checkpoint.pth')

    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f'All checkpoints founded in {output_dir}: {checkpoints}')
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f'The latest checkpoint founded: {latest_checkpoint}')
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
