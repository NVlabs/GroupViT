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
