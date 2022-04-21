# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

from mmcv.utils import Registry
from omegaconf import OmegaConf

MODELS = Registry('model')


def build_model(config):

    model = MODELS.build(OmegaConf.to_container(config, resolve=True))

    return model
