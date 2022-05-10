# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
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
