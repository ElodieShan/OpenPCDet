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
        eval_type = sys.argv[3]
        print("config_file:",config_file)
        print("result_pkl_file:",result_pkl_file)
        print("eval_type:",eval_type)

        dataset, dataset_type, cfg = load_dataset_from_openpcdet(config_file, train_mode=False)
        with open(result_pkl_file, 'rb') as f:
            det_annos = pickle.load(f)
        
        if eval_type == "evaluation":
            result_str, result_dict = dataset.evaluation(
                det_annos, dataset.class_names,
                eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
            )
            if 'min_thresh_ret' in result_dict:
                cls_recall = result_dict['min_thresh_ret']['recall_min_thresh']*100
                cls_precision = result_dict['min_thresh_ret']['precision_min_thresh']*100
                overlap = np.array([[0.0,0.0,0.0],[0.5,0.25,0.25],[0.7,0.5,0.5]])
                for m, current_class in enumerate(cfg.CLASS_NAMES):
                    print(current_class)
                    print("              Easy     Mod      Hard")
                    print("recall@%.1f:   %.2f    %.2f    %.2f"%(overlap[0,m], cls_recall[m,0,2], cls_recall[m,1,2], cls_recall[m,2,2]))
                    print("precison@%.1f: %.2f    %.2f    %.2f \n"%(overlap[0,m],cls_precision[m,0,2], cls_precision[m,1,2], cls_precision[m,2,2]))
                    print("recall@%.1f:   %.2f    %.2f    %.2f"%(overlap[1,m],cls_recall[m,0,1], cls_recall[m,1,1], cls_recall[m,2,1]))
                    print("precison@%.1f: %.2f    %.2f    %.2f \n"%(overlap[1,m],cls_precision[m,0,1], cls_precision[m,1,1], cls_precision[m,2,1]))
                    print("recall@%.1f:   %.2f    %.2f    %.2f"%(overlap[2,m],cls_recall[m,0,0], cls_recall[m,1,0], cls_recall[m,2,0]))
                    print("precison@%.1f: %.2f    %.2f    %.2f \n"%(overlap[2,m],cls_precision[m,0,0], cls_precision[m,1,0], cls_precision[m,2,0]))
            print('-------------\n')
            print(result_str)


        if eval_type == "iou":
            det_w_iou = dataset.get_detobject_iou(det_annos)
            result_w_iou_pkl_file= result_pkl_file.replace('result.pkl', 'result_w_iou.pkl')
            with open(result_w_iou_pkl_file,"wb") as f:
                pickle.dump(det_w_iou,f)
        # print(result_str)
