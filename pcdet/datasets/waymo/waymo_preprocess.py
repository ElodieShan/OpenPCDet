# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.


import os
import pickle
import numpy as np
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2
from tqdm import tqdm
from pathlib import Path
from pcdet.utils import common_utils

WAYMO_CLASSES = ['Vehicle', 'Pedestrian', 'Cyclist']

def create_obj_id_dict(gt_info_path):
    with open(gt_info_path, "rb") as f:
        gt_info_dict = pickle.load(f)

    # key: obj_id, value: info list 

    obj_id_dict = {
        'Vehicle': {},
        'Pedestrian': {},
        'Cyclist': {},
    }

    for class_name in WAYMO_CLASSES:
        gt_infos = gt_info_dict[class_name]
        print("Class - ", class_name, " : ", len(gt_infos))
        for i in tqdm(range(len(gt_infos))):
            gt_info = gt_infos[i]
            obj_id = gt_info['obj_id']
            if obj_id not in obj_id_dict[class_name]:
                obj_id_dict[class_name][obj_id] = [gt_info]
            else:
                obj_id_dict[class_name][obj_id].append(gt_info)
    
    obj_id_dict_path = gt_info_path.replace('.pkl','_obj_id.pkl')

    with open(obj_id_dict_path, 'wb') as f:
        pickle.dump(obj_id_dict, f)

def analyze_gt_dataset(gt_obj_id_info_path, point_feature_num=5, sample_num=1000):
    """
    Class -  Vehicle  :  46692
    num objs in 0-15m: (530171,)
    avg points: 3355.186296496791
    max points: 68432
    min points: 0

    Class -  Pedestrian  :  18469
    num objs in 0-15m: (297839,)
    avg points: 368.0195038258925
    max points: 6833
    min points: 0

    Class -  Cyclist  :  505
    num objs in 0-15m: (7234,)
    avg points: 679.6761128006635
    max points: 3024
    min points: 0
    """
    with open(gt_obj_id_info_path, 'rb') as f:
        obj_id_dict = pickle.load(f)
    for class_name in WAYMO_CLASSES:
        obj_gt_per_class_list = obj_id_dict[class_name]
        obj_ids = list(obj_gt_per_class_list.keys())
        print("\nClass - ", class_name, " : ", len(obj_ids))
        num_gt_points_0_15 = []
        for obj_id_key in tqdm(obj_ids):
            gt_list_per_obj = obj_gt_per_class_list[obj_id_key]
            for gt_info in gt_list_per_obj:
                box3d_lidar = gt_info['box3d_lidar']
                distance_box = np.sqrt(np.sum([box3d_lidar[0]**2,box3d_lidar[1]**2]))
                if distance_box < 15:
                    num_gt_points_0_15.append(gt_info['num_points_in_gt'])
        num_gt_points_0_15 = np.array(num_gt_points_0_15)
        print("num objs in 0-15m:",num_gt_points_0_15.shape)
        print("avg points:",np.mean(num_gt_points_0_15))    
        print("max points:",np.max(num_gt_points_0_15))    
        print("min points:",np.min(num_gt_points_0_15))    

def create_complished_gt_dataset(gt_obj_id_info_path, point_feature_num=5, sample_num=1000):
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    gt_info_root = ROOT_DIR / 'data' / 'waymo'
    if sample_num>0:
        dirname = 'pcdet_completed_gt_database_train_sample_%s' % (sample_num)
        database_save_path = gt_info_root / dirname
        print("database_save_path:",database_save_path)
    else:
        database_save_path = gt_info_root / ( 'pcdet_completed_gt_database_train' )
    database_save_path.mkdir(parents=True, exist_ok=True)

    with open(gt_obj_id_info_path, 'rb') as f:
        obj_id_dict = pickle.load(f)
    print("Load obj_id_dict from ", gt_obj_id_info_path)
    for class_name in WAYMO_CLASSES:
        obj_gt_per_class_list = obj_id_dict[class_name]
        obj_ids = list(obj_gt_per_class_list.keys())
        print("Class - ", class_name, " : ", len(obj_ids))
        for obj_id_key in tqdm(obj_ids):
            gt_list_per_obj = obj_gt_per_class_list[obj_id_key]
            completed_gt_points = None
            for gt_info in gt_list_per_obj:
                points_path = gt_info_root / gt_info["path"]
                box3d_lidar = gt_info['box3d_lidar']
                points_singe_frame = np.fromfile(points_path,dtype=np.float32).reshape([-1, point_feature_num ])
                points_singe_frame = common_utils.rotate_points_along_z(points_singe_frame[np.newaxis, :, :], np.array([-box3d_lidar[-1]]))[0]
                if completed_gt_points is None:
                    completed_gt_points = points_singe_frame
                else:
                    completed_gt_points = np.vstack((completed_gt_points, points_singe_frame))
            if sample_num>0:
                if sample_num < completed_gt_points.shape[0]:
                    choice = np.arange(0, completed_gt_points.shape[0], dtype=np.int32)
                    choice = np.random.choice(choice, sample_num, replace=False)
                    completed_gt_points = completed_gt_points[choice]
            filename = '%s.bin' % (obj_id_key)
            filepath = database_save_path / filename
            with open(filepath, 'w') as f:
                completed_gt_points.tofile(f)   

def sample_complished_gt_dataset(gt_obj_id_info_path, point_feature_num=5, sample_num=1000):
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    gt_info_root = ROOT_DIR / 'data' / 'waymo'
    if sample_num>0:
        dirname = 'pcdet_completed_gt_database_train_sample_%s' % (sample_num)
        database_save_path = gt_info_root / dirname
        database_ori_path = gt_info_root / ( 'pcdet_completed_gt_database_train' )
        print("database_ori_path:",database_ori_path)
        print("database_save_path:",database_save_path)
    else:
        database_save_path = gt_info_root / ( 'pcdet_completed_gt_database_train' )
    database_save_path.mkdir(parents=True, exist_ok=True)

    with open(gt_obj_id_info_path, 'rb') as f:
        obj_id_dict = pickle.load(f)
    print("Load obj_id_dict from ", gt_obj_id_info_path)
    for class_name in WAYMO_CLASSES:
        obj_gt_per_class_list = obj_id_dict[class_name]
        obj_ids = list(obj_gt_per_class_list.keys())
        print("Class - ", class_name, " : ", len(obj_ids))
        for obj_id_key in tqdm(obj_ids):
            filename_completed = '%s.bin' % (obj_id_key)
            filepath_completed = database_ori_path / filename_completed
            completed_gt_points=np.fromfile(filepath_completed,dtype=np.float32).reshape([-1, point_feature_num ])  
            if sample_num>0:
                if sample_num < completed_gt_points.shape[0]:
                    choice = np.arange(0, completed_gt_points.shape[0], dtype=np.int32)
                    choice = np.random.choice(choice, sample_num, replace=False)
                    completed_gt_points = completed_gt_points[choice]

            filename = '%s.bin' % (obj_id_key)
            filepath = database_save_path / filename
            with open(filepath, 'w') as f:
                completed_gt_points.tofile(f)   


if __name__ == '__main__':
    # gt_info_path = "/home/elodie/OpenPCDet/data/waymo/pcdet_waymo_dbinfos_train_sampled_1.pkl"
    # create_obj_id_dict(gt_info_path)
    gt_obj_id_info_path = "/home/elodie/OpenPCDet/data/waymo/pcdet_waymo_dbinfos_train_sampled_1_obj_id.pkl"
    sample_complished_gt_dataset(gt_obj_id_info_path)
    # create_complished_gt_dataset(gt_obj_id_info_path)
    # analyze_gt_dataset(gt_obj_id_info_path)