#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

CFG_DIR=../output/kitti_models/demo
CFG_FILE=VLP-pv_rcnn_mimic-TPv1.yaml
EPOCH=80
TAG=test_efficiency_16linesTPv1-batch2

CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 test.py --launcher pytorch \
--cfg_file cfgs/kitti_models/second.yaml \
--output_dir $CFG_DIR \
--eval_tag $TAG \
--tcp_port 18802 \
--ckpt ../models/demo/second_7862.pth \