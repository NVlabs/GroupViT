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

from mmseg.datasets import DATASETS
from mmseg.datasets import PascalContextDataset as _PascalContextDataset


@DATASETS.register_module(force=True)
class PascalContextDataset(_PascalContextDataset):

    CLASSES = ('background', 'airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book',
               'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow',
               'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse',
               'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'plant', 'road',
               'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree',
               'truck', 'monitor', 'wall', 'water', 'window', 'wood')
