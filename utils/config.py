# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import os
import os.path as osp

from omegaconf import OmegaConf


def load_config(cfg_file):
    cfg = OmegaConf.load(cfg_file)
    if '_base_' in cfg:
        if isinstance(cfg._base_, str):
            base_cfg = OmegaConf.load(osp.join(osp.dirname(cfg_file), cfg._base_))
        else:
            base_cfg = OmegaConf.merge(OmegaConf.load(f) for f in cfg._base_)
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg


def get_config(args):
    cfg = load_config(args.cfg)
    OmegaConf.set_struct(cfg, True)

    if args.opts is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.opts))
    if hasattr(args, 'batch_size') and args.batch_size:
        cfg.data.batch_size = args.batch_size

    if hasattr(args, 'amp_opt_level') and args.amp_opt_level:
        cfg.train.amp_opt_level = args.amp_opt_level

    if hasattr(args, 'resume') and args.resume:
        cfg.checkpoint.resume = args.resume

    if hasattr(args, 'eval') and args.eval:
        cfg.evaluate.eval_only = args.eval

    if hasattr(args, 'keep') and args.keep:
        cfg.checkpoint.max_kept = args.keep

    if not cfg.model_name:
        cfg.model_name = osp.splitext(osp.basename(args.cfg))[0]

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    cfg.model_name = cfg.model_name + f'_bs{cfg.data.batch_size}x{world_size}'

    if hasattr(args, 'output') and args.output:
        cfg.output = args.output
    else:
        cfg.output = osp.join('output', cfg.model_name)

    if hasattr(args, 'tag') and args.tag:
        cfg.tag = args.tag
        cfg.output = osp.join(cfg.output, cfg.tag)

    if hasattr(args, 'wandb') and args.wandb:
        cfg.wandb = args.wandb

    if hasattr(args, 'vis') and args.vis:
        cfg.vis = args.vis

    cfg.local_rank = args.local_rank

    OmegaConf.set_readonly(cfg, True)

    return cfg
