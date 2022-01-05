# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.


import os
import pickle
import numpy as np
from pathlib import Path
from pcdet.utils import common_utils, box_utils
from pcdet.utils import pointcloud_utils
from pcdet.utils import pointcloud_sample_utils
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2

WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']


def generate_labels(frame):
    pass

def get_points(data_path, sequence_name, sample_idx):
    lidar_file = data_path / sequence_name / ('%04d.npy' % sample_idx)
    point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]
    points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
    points_all = points_all[NLZ_flag == -1]
    points_ring = point_features[:, 6]
    points_ring = points_ring[NLZ_flag == -1]
    return points_all, points_ring

if __name__ == '__main__':
    root_path = Path('/home/elodie/OpenPCDet/data/waymo_ring/')
    val_info_path = root_path / 'waymo_ring_processed_data_v0_5_0_infos_val.pkl'
    data_path = root_path / 'waymo_ring_processed_data_v0_5_0'
    save_val_info_path = root_path / 'waymo_ring_processed_data_v0_5_0_infos_val_sampled_gtnum_add64.pkl'
    with open(val_info_path, 'rb') as f:
        val_info = pickle.load(f)
    
    for idx, info in enumerate(val_info):
        print(idx, len(val_info))
        # print(info['point_cloud'].keys())
        # print('   ',info['annos'].keys())
        pc_info = info['point_cloud']
        points, ring = get_points(data_path, pc_info['lidar_sequence'], pc_info['sample_idx'])
        # print(points.shape)
        # val_info[idx]['annos']['num_points_in_gt_sampled'] = {}
        anno_info = info['annos']
        anno_info['num_points_in_gt_sampled'] = {}
        num_points_in_gt_ori = anno_info['num_points_in_gt']
        for sample_type in ['Waymo_v1', 'Waymo_v2', 'Waymo_v3', 'Waymo_64']:
            points_sampled = pointcloud_sample_utils.downsample_waymo(points, ring, sample_type, verticle_switch=True)
            gt_boxes_lidar = anno_info['gt_boxes_lidar']
    
            num_objects = gt_boxes_lidar.shape[0]
            num_gt = len(anno_info['name'])
            corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
            num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

            for k in range(num_objects):
                flag = box_utils.in_hull(points_sampled[:, 0:3], corners_lidar[k])
                num_points_in_gt[k] = flag.sum()
            anno_info['num_points_in_gt_sampled'][sample_type] = num_points_in_gt
        # print(num_points_in_gt_ori,val_info[idx]['annos']['num_points_in_gt_sampled'])
    print("Dump")
    with open(save_val_info_path, 'wb') as f:
        pickle.dump(val_info, f)