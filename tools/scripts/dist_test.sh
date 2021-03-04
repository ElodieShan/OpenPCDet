#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

# CFG_DIR=../output/audi_models/second_16lines_audi/21030202-Audi-80epoch-batch4
# CFG_FILE=second_16lines_audi.yaml
# EPOCH=80
# TAG=test_gt_box

CFG_DIR=../output/kitti_models/second_16lines_vlp/second_16lines_vlp-2
CFG_FILE=second_16lines_vlp.yaml
EPOCH=16
TAG=test
# TAG=test_gt_lhw_yzx_r_h2_gt

# TAG=test_lhw_yzx_r_h2

CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 test.py --launcher pytorch \
--cfg_file $CFG_DIR/$CFG_FILE \
--output_dir $CFG_DIR \
--batch_size 2 \
--eval_tag $TAG \
--tcp_port 18882 \
--ckpt $CFG_DIR/ckpt/checkpoint_epoch_$EPOCH.pth \

######### Test Teacher
# CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 test_teacher.py --launcher pytorch \
# --cfg_file $CFG_DIR/$CFG_FILE \
# --output_dir $CFG_DIR \
# --batch_size 2 \
# --eval_tag $TAG \
# --tcp_port 18881 \
# --ckpt $CFG_DIR/ckpt/checkpoint_epoch_$EPOCH.pth \
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
