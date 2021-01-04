#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
# --nproc_per_node=2 train.py --launcher pytorch \
# --cfg_file cfgs/kitti_models/second2.yaml \
# --extra_tag second_sub \
# --ckpt_save_interval 2 \
# --use_sub_data

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 train.py \
--launcher pytorch --cfg_file cfgs/kitti_models/second_cross.yaml \
--tcp_port 18888 \
--extra_tag test-second_cross \
--ckpt_save_interval 2 \
--cross_sample_prob 0.5

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 train.py \
# --launcher pytorch --cfg_file cfgs/kitti_models/second_spconv_v2.yaml \
# --tcp_port 18878 \
# --extra_tag second_16lines_spconv_v2 \
# --ckpt_save_interval 2 



# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
# --nproc_per_node=2 train.py --launcher pytorch \
# --cfg_file cfgs/kitti_models/second_spconv_v3.yaml \
# --extra_tag VoxelBackBone8x_v3 \
# --ckpt_save_interval 2

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
# --nproc_per_node=2 train.py --launcher pytorch \
# --cfg_file cfgs/kitti_models/second2.yaml \
# --extra_tag plane_lr0.003

