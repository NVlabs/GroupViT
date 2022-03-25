# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual
# property and proprietary rights in and to this software, related
# documentation and any modifications thereto.  Any use, reproduction,
# disclosure or distribution of this software and related documentation
# without an express license agreement from NVIDIA CORPORATION is strictly
# prohibited.
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.epochs * n_iter_per_epoch)
    warmup_steps = int(config.warmup_epochs * n_iter_per_epoch)

    lr_scheduler = None
    if config.lr_scheduler.name == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=config.min_lr,
            warmup_lr_init=config.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        raise NotImplementedError(f'lr scheduler {config.lr_scheduler.name} not implemented')

    return lr_scheduler
