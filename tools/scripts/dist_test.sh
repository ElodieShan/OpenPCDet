#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

# CFG_DIR=../output/kitti_models/second_16lines/use_plane_batch2_lr0.0015
# CFG_FILE=second_16lines.yaml
# EPOCH=80
# TAG=eval_second_16lines

# CFG_DIR=../output/kitti_models/second/use_plane_batch2_lr0.0015
# CFG_FILE=second2.yaml
# EPOCH=80
# TAG=eval_second

# CFG_DIR=../output/kitti_models/second_mimic_MSE2_BoundedReg3/pretrained_mse_ttp0_2_sfp5_breg_a3_m00001
# CFG_FILE=second_mimic_MSE2_BoundedReg3_2.yaml
# EPOCH=40
# TAG=pretrained_mse_ttp0_2_sfp5_breg_a3_m00001

# CFG_DIR=../output/kitti_models/second_mimic_MSE_BoundedReg/pretrained_mse2_ttp0_2_sfp5_reg_a3m1e_5_hintl2_0_3_ttp1_sfp0_5
# CFG_FILE=second_mimic_MSE_BoundedReg_3.yaml
# EPOCH=40
# TAG=pretrained_mse2_ttp0_2_sfp5_reg_a3m1e_5_hintl2_0_3_ttp1_sfp0_5

CFG_DIR=../output/kitti_models/second/SoftmaxFocalClassificationLoss1
CFG_FILE=second.yaml
EPOCH=80
TAG=pr_eval

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test.py --launcher pytorch \
--cfg_file $CFG_DIR/$CFG_FILE \
--batch_size 2 \
--eval_tag $TAG \
--tcp_port 18881 \
--ckpt $CFG_DIR/ckpt/checkpoint_epoch_$EPOCH.pth

# --ckpt_dir $CFG_DIR/ckpt/ \
# --eval_all



# --save_iou
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test.py --launcher pytorch \
# --cfg_file cfgs/kitti_models/second_mimic_test.yaml \
# --batch_size 2 \
# --eval_tag pretained-MSE2_thc0.2_sfp5-HintL20.5_thc1_tsp_1 \
# --tcp_port 18881 \
# --ckpt ../output/kitti_models/second_mimic_MSE_Hint/pretained-MSE2_thc0.2_sfp5-HintL20.5_thc1_tsp_1/ckpt/checkpoint_epoch_40.pth
