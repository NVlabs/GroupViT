#!/usr/bin/env bash

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


SCRIPT=$1
CONFIG=$2
NODE_RNAK=$3
NODES=$4
GPUS_PER_NODE=$5
MASTER_ADDR=$6
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
  --nnodes=$NODES --node_rank=$NODE_RNAK \
  --master_addr=$MASTER_ADDR  \
    $SCRIPT --cfg $CONFIG ${@:7}
