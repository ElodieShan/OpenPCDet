#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

CFG_DIR=../output/kitti_models/second_16lines_tp16_v3-1/TPv3-30epoch-batch4-resume_by_kl20_gt10_sfp40_onlyt-regv2_1_m1e-5_gt_1-2
CFG_FILE=second_16lines_tp16_v3-1.yaml
EPOCH=80
TAG=test_16lines_batch4

CUDA_VISIBLE_DEVICES=5 python3 -m torch.distributed.launch --nproc_per_node=1 test.py --launcher pytorch \
--cfg_file $CFG_DIR/$CFG_FILE \
--output_dir $CFG_DIR \
--batch_size 4 \
--eval_tag $TAG \
--tcp_port 18881 \
--ckpt $CFG_DIR/ckpt/checkpoint_epoch_$EPOCH.pth 

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
