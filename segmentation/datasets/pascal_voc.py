# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

from mmseg.datasets import DATASETS
from mmseg.datasets import PascalVOCDataset as _PascalVOCDataset


@DATASETS.register_module(force=True)
class PascalVOCDataset(_PascalVOCDataset):

    CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'table', 'dog', 'horse', 'motorbike', 'person', 'plant', 'sheep', 'sofa', 'train', 'monitor')
