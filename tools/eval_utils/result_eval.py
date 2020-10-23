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

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'MultipleDataset': MultipleDataset
}
def get_logger(log_level=logging.INFO, log_path=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    if log_path is None:
        import datetime
        log_path = ("/home/elodie/det3d_ros/src/det3d_pcl/log/%s" % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def load_dataset_from_openpcdet(config_path, train_mode=True):
    from pcdet.config import cfg, cfg_from_yaml_file
    cfg = cfg_from_yaml_file(config_path, cfg)
    logger = get_logger(log_level=logging.INFO)
    dataset = __all__[cfg.DATA_CONFIG.DATASET](
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=train_mode,
        logger=logger,
    )
    return dataset, dataset.dataset_cfg.DATASET, cfg

if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1:
        import yaml
        from tqdm import tqdm
        from easydict import EasyDict
        config_file = sys.argv[1]
        result_pkl_file = sys.argv[2]
        dataset, dataset_type, cfg = load_dataset_from_openpcdet(config_file, train_mode=False)
        with open(result_pkl_file, 'rb') as f:
            det_annos = pickle.load(f)

        result_str, result_dict = dataset.evaluation(
            det_annos, dataset.class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
        )

        print(result_str)
