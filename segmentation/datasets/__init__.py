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

from .coco_object import COCOObjectDataset
from .pascal_context import PascalContextDataset
from .pascal_voc import PascalVOCDataset

__all__ = ['COCOObjectDataset', 'PascalContextDataset', 'PascalVOCDataset']
