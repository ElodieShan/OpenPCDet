#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

CFG_DIR=../output/kitti_models/second/use_plane_batch2_lr0.0015
CFG_FILE=second2.yaml
EPOCH=80
TAG=test_64lines_batch4

CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 test.py --launcher pytorch \
--cfg_file $CFG_DIR/$CFG_FILE \
--output_dir $CFG_DIR \
--batch_size 4 \
--eval_tag $TAG \
--tcp_port 18881 \
--ckpt $CFG_DIR/ckpt/checkpoint_epoch_$EPOCH.pth 

# --ckpt $CFG_DIR/ckpt/checkpoint_epoch_$EPOCH.pth \


# CFG_DIR=../output/kitti_models/tsubv1_second_mimic/teachersubv1-ResBB-respretrained_klv2_20_gt10_sfp40_onlyt-2
# CFG_FILE=second_mimic_KL_2.yaml
# EPOCH=40
# TAG=test_on_16lines

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test.py --launcher pytorch \
# --cfg_file $CFG_DIR/$CFG_FILE \
# --output_dir $CFG_DIR \
# --batch_size 2 \
# --eval_tag $TAG \
# --tcp_port 18881 \
# --ckpt $CFG_DIR/ckpt/checkpoint_epoch_$EPOCH.pth \
# --use_sub_data
