#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
--nproc_per_node=2 train.py --launcher pytorch \
--cfg_file cfgs/kitti_models/second.yaml \
--extra_tag plane_lr0.0015

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
--nproc_per_node=2 train.py --launcher pytorch \
--cfg_file cfgs/kitti_models/second2.yaml \
--extra_tag plane_lr0.003

