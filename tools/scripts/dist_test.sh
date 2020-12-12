#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}


CFG_DIR=../output/kitti_models/second2/second_sub
CFG_FILE=second2.yaml
EPOCH=80
TAG=test_on_64lines3

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test.py --launcher pytorch \
--cfg_file $CFG_DIR/$CFG_FILE \
--output_dir $CFG_DIR \
--batch_size 2 \
--eval_tag $TAG \
--tcp_port 18881 \
--ckpt $CFG_DIR/ckpt/checkpoint_epoch_$EPOCH.pth \
--use_sub_data
