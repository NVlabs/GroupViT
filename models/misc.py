# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import math

import torch.nn.functional as F


class Result:

    def __init__(self, as_dict=False):
        if as_dict:
            self.outs = {}
        else:
            self.outs = []

    @property
    def as_dict(self):
        return isinstance(self.outs, dict)

    def append(self, element, name=None):
        if self.as_dict:
            assert name is not None
            self.outs[name] = element
        else:
            self.outs.append(element)

    def update(self, **kwargs):
        if self.as_dict:
            self.outs.update(**kwargs)
        else:
            for v in kwargs.values():
                self.outs.append(v)

    def as_output(self):
        if self.as_dict:
            return self.outs
        else:
            return tuple(self.outs)

    def as_return(self):
        outs = self.as_output()
        if self.as_dict:
            return outs
        if len(outs) == 1:
            return outs[0]
        return outs


def interpolate_pos_encoding(pos_embed, H, W):
    num_patches = H * W

    N = pos_embed.shape[1]
    if num_patches == N and W == H:
        return pos_embed
    patch_pos_embed = pos_embed
    dim = pos_embed.shape[-1]
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        size=(H, W),
        mode='bicubic',
        align_corners=False)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return patch_pos_embed
