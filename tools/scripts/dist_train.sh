#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
# --nproc_per_node=2 train.py --launcher pytorch \
# --cfg_file cfgs/audi_models/second_audi2.yaml \
# --extra_tag test \
# --ckpt_save_interval 2

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
--nproc_per_node=2 train.py --launcher pytorch \
--cfg_file cfgs/audi_models/second_16lines_audi4.yaml \
--extra_tag 21031003-Audi-mode4-16lines-80epoch-batch4-mode3 \
--tcp_port 18811 \
--ckpt_save_interval 2

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
--nproc_per_node=2 train.py --launcher pytorch \
--cfg_file cfgs/audi_models/second_16lines_audi4.yaml \
--extra_tag 21031004-Audi-mode4-16lines-80epoch-batch4-mode3 \
--tcp_port 18811 \
--ckpt_save_interval 2


# CUDA_VISIBLE_DEVICES=6,7 python3 -m torch.distributed.launch \
# --nproc_per_node=2 train.py --launcher pytorch \
# --cfg_file cfgs/kitti_models/pv_rcnn_16lines.yaml \
# --extra_tag 21030206-VLP-80epoch-batch2 \
# --ckpt_save_interval 2

# CUDA_VISIBLE_DEVICES=6,7 python3 -m torch.distributed.launch \
# --nproc_per_node=2 train.py --launcher pytorch \
# --cfg_file cfgs/kitti_models/pv_rcnn_16lines.yaml \
# --extra_tag 21030207-VLP-80epoch-batch2 \
# --ckpt_save_interval 2

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
# --nproc_per_node=2 train.py --launcher pytorch \
# --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
# --extra_tag 16lines-w-planes-batch1 \
# --ckpt_save_interval 2 \
# --tcp_port 18875

# CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch \
# --nproc_per_node=2 train.py --launcher pytorch \
# --cfg_file cfgs/kitti_models/second_16lines_tp16_v3-1.yaml \
# --ckpt ../output/kitti_models/second_16lines_tp16_v3-mimic2/TPv3-80epoch-batch4-kl20_gt10_sfp40_onlyt-regv2_1_m1e-5_gt_1-2/ckpt/checkpoint_epoch_51.pth \
# --tcp_port 18898 \
# --extra_tag  TPv3-30epoch-batch4-resume_by_kl20_gt10_sfp40_onlyt-regv2_1_m1e-5_gt_1-2

# CUDA_VISIBLE_DEVICES=6,7 python3 -m torch.distributed.launch \
# --nproc_per_node=2 train.py --launcher pytorch \
# --cfg_file cfgs/kitti_models/second_16lines_tp16_v3-1.yaml \
# --ckpt ../output/kitti_models/second_16lines_tp16_v3-mimic2/TPv3-80epoch-batch4-kl20_gt10_sfp40_onlyt-regv2_1_m1e-5_gt_1-2/ckpt/checkpoint_epoch_51.pth \
# --tcp_port 18898 \
# --extra_tag  TPv3-30epoch-batch4-resume_by_kl20_gt10_sfp40_onlyt-regv2_1_m1e-5_gt_1-2-2

# CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch \
# --nproc_per_node=2 train.py --launcher pytorch \
# --cfg_file cfgs/kitti_models/second_16lines_tp16_v2.yaml \
# --extra_tag 16lines-w-planes-batch4 \
# --ckpt_save_interval 2 \
# --tcp_port 18875



# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
# --nproc_per_node=2 train.py --launcher pytorch \
# --cfg_file cfgs/kitti_models/second_spconv_v3.yaml \
# --extra_tag VoxelBackBone8x_v3 \
# --ckpt_save_interval 2

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
# --nproc_per_node=2 train.py --launcher pytorch \
# --cfg_file cfgs/kitti_models/second2.yaml \
# --extra_tag plane_lr0.003

