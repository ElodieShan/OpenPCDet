#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}

CUDA_VISIBLE_DEVICES=1,2 python3 -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch \
--cfg_file cfgs/kitti_models/second_16lines.yaml \
--pretrained_model ../output/kitti_models/second/default/ckpt/checkpoint_epoch_80.pth \
--extra_tag fine_tune_lr0.001 \
--tcp_port 18887