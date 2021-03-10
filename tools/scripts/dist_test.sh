#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

CFG_DIR=../output/audi_models/second_audi4_mimic/21031001-Audi-mode4-80epoch-batch4-CLS_klv2_20_gt10_sf50_onlyt_DebugFN-REG_v2_a1m1e5_gt1
CFG_FILE=second_audi4_mimic.yaml
EPOCH=80
TAG=test_123

CUDA_VISIBLE_DEVICES=4 python3 -m torch.distributed.launch --nproc_per_node=1 test.py --launcher pytorch \
--cfg_file $CFG_DIR/$CFG_FILE \
--output_dir $CFG_DIR \
--batch_size 4 \
--eval_tag $TAG \
--tcp_port 18831 \
--ckpt $CFG_DIR/ckpt/checkpoint_epoch_$EPOCH.pth

# CUDA_VISIBLE_DEVICES=4 python3 -m torch.distributed.launch --nproc_per_node=1 test.py --launcher pytorch \
# --cfg_file $CFG_DIR/$CFG_FILE \
# --output_dir $CFG_DIR \
# --batch_size 4 \
# --eval_tag $TAG \
# --tcp_port 18831 \
# --ckpt_dir $CFG_DIR \
# --eval_all

# --use_sub_data

# --ckpt ../output/kitti_models/pointpillar/demo/pointpillar_7728.pth



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
