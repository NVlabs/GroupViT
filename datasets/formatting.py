# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import torch
from mmcv.parallel import DataContainer as DC


class ToDataContainer(object):
    """Convert results to :obj:`mmcv.DataContainer`"""

    def __call__(self, sample):
        """Call function to convert data in results to
        :obj:`mmcv.DataContainer`.

        Args:
            sample (torch.Tensor): Input sample.

        Returns:
            DataContainer
        """
        if isinstance(sample, int):
            sample = torch.tensor(sample)
        return DC(sample, stack=True, pad_dims=None)

    def __repr__(self):
        return self.__class__.__name__
