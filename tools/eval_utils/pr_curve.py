import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.utils import common_utils

import logging
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.datasets.multiple.multiple_dataset import MultipleDataset
from pcdet.datasets import build_dataloader
from pcdet.utils.box_utils import boxes_to_corners_3d

import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'MultipleDataset': MultipleDataset
}

def init_data_dir(path):
    import os
    file_root_path, file_name=os.path.split(path)
    if not os.path.exists(file_root_path):
        try:
            os.makedirs(file_root_path)
        except:
            print("Error occurs when make dirs %s"%file_root_path)

def get_logger(log_level=logging.INFO, log_dir=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    if log_dir is None:
        log_path = ("/home/elodie/det3d_ros/src/det3d_pcl/log/pr_curve-%s.log" % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    else:
        log_path = ("%s/pr_curve-%s.log" % (log_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
        init_data_dir(log_path)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

# precision[3,41]/[3,11] easy,mod,hard
def plot_pr_curve(precision, precision_mimic, out_pic_fig_path):
    plt.figure()
    recall = np.linspace(0,1,41)
    plt.plot(recall, precision['easy'], color='darkorange', label='Easy-Ori') 
    plt.plot(recall, precision['mod'], color='b', label='Mod-Ori') 
    plt.plot(recall, precision['hard'], color='g', label='Hard-Ori') 

    plt.plot(recall, precision_mimic['easy'], '--', color='darkorange', label='Easy-Mimic') 
    plt.plot(recall, precision_mimic['mod'],  '--', color='b', label='Mod-Mimic') 
    plt.plot(recall, precision_mimic['hard'], '--', color='g',  label='Hard-Mimic') 

    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_pic_fig_path)
    plt.clf() 

def plot_pr_curve_v2(precision, precision_mimic, out_pic_fig_path, difficulty='easy'):
    plt.figure()
    recall = np.linspace(0,1,41)
    plt.plot(recall, precision['car'][difficulty], color='darkorange', label='car-Ori') 
    plt.plot(recall, precision['pedestrian'][difficulty], color='b', label='pedestrian-Ori') 
    plt.plot(recall, precision['cyclist'][difficulty], color='g', label='cyclist-Ori') 

    plt.plot(recall, precision_mimic['car'][difficulty], '--', color='darkorange', label='car-Mimic') 
    plt.plot(recall, precision_mimic['pedestrian'][difficulty],  '--', color='b', label='pedestrian-Mimic') 
    plt.plot(recall, precision_mimic['cyclist'][difficulty], '--', color='g',  label='cyclist-Mimic') 

    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_pic_fig_path)
    plt.clf() 


def get_precision_dict(result_dict, ret_type='3d'):
    PR_detail_dict = result_dict['PR_detail_dict'][ret_type]['precision']
    precision_dict = {
        'car':{
            'easy':PR_detail_dict[0,0,0],
            'mod':PR_detail_dict[0,1,0],
            'hard':PR_detail_dict[0,2,0],
        },
        'pedestrian':{
            'easy':PR_detail_dict[1,0,0],
            'mod':PR_detail_dict[1,1,0],
            'hard':PR_detail_dict[1,2,0],
        },
        'cyclist':{
            'easy':PR_detail_dict[2,0,0],
            'mod':PR_detail_dict[2,1,0],
            'hard':PR_detail_dict[2,2,0],
        }
    }
    return precision_dict

if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1:
        import yaml
        from tqdm import tqdm
        from easydict import EasyDict
        config_dir_ori = sys.argv[1]
        config_file_ori = sys.argv[2]
        result_dict_pkl_file_ori = sys.argv[3]
        config_dir_mimic = sys.argv[4]
        result_dict_pkl_file_mimic = sys.argv[5]

        log_dir = config_dir_mimic + "/result_eval"
        logger = get_logger(log_level=logging.INFO, log_dir=log_dir)
        with open(result_dict_pkl_file_ori, 'rb') as f:
            result_dict_ori = pickle.load(f)

        with open(result_dict_pkl_file_mimic, 'rb') as f:
            result_dict_mimic = pickle.load(f)
        classes =['car','pedestrian','cyclist']
        if result_dict_mimic is not None:
            logger.info("Tips: ori means not use mimic..\n")

            logger.info("config_file: %s"%config_file_ori)

            logger.info("config_dir_ori: %s"%config_dir_ori)
            logger.info("result_dict_pkl_file_ori: %s"%result_dict_pkl_file_ori)
            logger.info("config_dir_mimic: %s"%config_dir_mimic)
            logger.info("result_dict_pkl_file_mimic: %s"%result_dict_pkl_file_mimic)

            assert 'PR_detail_dict' in result_dict_ori, "PR_detail_dict not in result_dict_ori!"
            assert 'PR_detail_dict' in result_dict_mimic, "PR_detail_dict not in result_dict_mimic!"

            ori_precision_dict_3d = get_precision_dict(result_dict_ori, ret_type='3d')
            mimic_precision_dict_3d = get_precision_dict(result_dict_mimic, ret_type='3d')
            output_file = log_dir + "/mimic_pr_curve-" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + ".png"
            plot_pr_curve(ori_precision_dict_3d['car'], mimic_precision_dict_3d['car'], out_pic_fig_path=output_file)
            output_file2 = log_dir + "/mimic_pr_curve-classes-" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + ".png"
            plot_pr_curve_v2(ori_precision_dict_3d, mimic_precision_dict_3d, out_pic_fig_path=output_file2)
            
            
