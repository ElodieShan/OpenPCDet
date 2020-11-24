#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
--nproc_per_node=2 train.py --launcher pytorch \
--cfg_file cfgs/kitti_models/second.yaml \
--extra_tag SoftmaxFocalClassificationLoss1-test \
--max_ckpt_save_num 30 \
--ckpt_save_interval 2 \
--pretrained_model ../output/kitti_models/second/SoftmaxFocalClassificationLoss1/ckpt/checkpoint_epoch_80.pth

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
# --nproc_per_node=2 train.py --launcher pytorch \
# --cfg_file cfgs/kitti_models/second2.yaml \
# --extra_tag plane_lr0.003

