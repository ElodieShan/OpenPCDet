#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}


CFG_DIR=../output/kitti_models/second/use_plane_batch2_lr0.0015
CFG_FILE=second_2.yaml
EPOCH=80
TAG=pr_eval_cls_thred0.5

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test.py --launcher pytorch \
--cfg_file $CFG_DIR/$CFG_FILE \
--output_dir $CFG_DIR \
--batch_size 2 \
--eval_tag $TAG \
--tcp_port 18881 \
--ckpt $CFG_DIR/ckpt/checkpoint_epoch_$EPOCH.pth
