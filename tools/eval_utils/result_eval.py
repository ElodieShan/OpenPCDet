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
    import datetime
    if log_dir is None:
        log_path = ("/home/elodie/det3d_ros/src/det3d_pcl/log/%s.log" % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    else:
        log_path = ("%s/%s.log" % (log_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
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

def load_dataset_from_openpcdet(config_path, train_mode=True, logger=None, log_dir=None):
    from pcdet.config import cfg, cfg_from_yaml_file
    cfg = cfg_from_yaml_file(config_path, cfg)

    if logger is None:
        logger = get_logger(log_level=logging.INFO, log_dir=log_dir)
    dataset = __all__[cfg.DATA_CONFIG.DATASET](
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=train_mode,
        logger=logger,
    )
    return dataset, dataset.dataset_cfg.DATASET, cfg, logger

# precision[3,41]/[3,11] easy,mod,hard
def plot_pr_curve(precision, out_pic_fig_path):
    plt.figure()
    recall = np.linspace(0,1,41)
    plt.plot(recall, precision['easy'], color='darkorange', label='Easy') 
    plt.plot(recall, precision['mod'], color='b', label='Mod') 
    plt.plot(recall, precision['hard'], color='g', label='Hard') 

    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_pic_fig_path)
    plt.clf() 

def plot_recall_iou(recall, precision, overlaps, out_pic_fig_path):
    pic_list = ['Easy', 'Mod', 'Hard']
    color = ['b','g','y','r']
    class_name = recall.keys()
    fig,axes = plt.subplots(nrows=2, ncols=len(pic_list), figsize=(16,9)) # row1: recall / row2:precision
    for i in range(len(pic_list)):
        for j, name in enumerate(class_name):
            print(j, name)
            print(recall[name][i])
            print(precision[name][i])
            if j == 3:
                overlap = np.sort(overlaps[:,0])
            else:
                overlap = np.sort(overlaps[:,j])
            axes[0][i].plot(overlap, recall[name][i], color=color[j], label=name)
            axes[1][i].plot(overlap, precision[name][i], color=color[j], label=name)
        axes[0][i].set_xticks(np.linspace(0,0.8,9))
        axes[0][i].set_yticks(np.linspace(50,100,6))
        title1 = 'Recall - ' + str(pic_list[i])
        axes[0][i].set_title(title1)
        axes[0][i].legend()
        axes[1][i].set_xticks(np.linspace(0,0.8,9))
        axes[1][i].set_yticks(np.linspace(0,60,7))
        title2 = 'Precision - ' + str(pic_list[i])
        axes[1][i].set_title(title2)
        axes[1][i].legend()
    # plt.show()
    fig.savefig(out_pic_fig_path)
    plt.clf() 

def convert_matrix2str(mat):
    str_mat = "["
    for i in mat:
        str_mat += str(i)
        str_mat += ", "
    str_mat = str_mat[:-2] + ']'
    return str_mat
    
if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1:
        import yaml
        from tqdm import tqdm
        from easydict import EasyDict
        config_dir = sys.argv[1]
        config_file = sys.argv[2]
        result_pkl_file = sys.argv[3]
        eval_type = sys.argv[4]
        log_dir = config_dir + "/result_eval"
        dataset, dataset_type, cfg, logger = load_dataset_from_openpcdet(config_file, train_mode=False, log_dir=log_dir)
        with open(result_pkl_file, 'rb') as f:
            det_annos = pickle.load(f)
        classes =['car','pedestrian','cyclist']
        if eval_type == "evaluation":
            result_str, result_dict = dataset.evaluation(
                det_annos, dataset.class_names,
                eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
            )

            logger.info("config_file: %s"%config_file)
            logger.info("result_pkl_file: %s"%result_pkl_file)
            logger.info("eval_type: %s"%eval_type)

            cls_precision_ = {
                'car': [],
                'pedestrian': [],
                'cyclist': [],
                'ignore_class': []
            }
            cls_recall_ = {
                'car': [],
                'pedestrian': [],
                'cyclist': [],
                'ignore_class': []
            }

            if 'min_thresh_ret' in result_dict:
                cls_recall = result_dict['min_thresh_ret']['recall_min_thresh']*100
                cls_precision = result_dict['min_thresh_ret']['precision_min_thresh']*100
                logger.info("\n==================== Precision & Recall Result =================")
                overlap = np.array([[0.7,0.5,0.5],
                                    [0.5,0.25,0.25],
                                    [0.4,0.4,0.4],
                                    [0.2,0.1,0.1],
                                    [0.0,0.0,0.0]])
                idx = np.array([[4,3,2,1,0],
                                [4,3,1,2,0],
                                [4,3,1,2,0]])
                for m, current_class in enumerate(cfg.CLASS_NAMES):
                    logger.info(current_class)
                    logger.info("              Easy     Mod      Hard")
                    for i in idx[m]:
                        logger.info("recall@%.1f:   %.2f    %.2f    %.2f"%(overlap[i,m], cls_recall[m,0,i], cls_recall[m,1,i], cls_recall[m,2,i]))
                        logger.info("precison@%.1f: %.2f    %.2f    %.2f \n"%(overlap[i,m],cls_precision[m,0,i], cls_precision[m,1,i], cls_precision[m,2,i]))
                    for j in range(len(classes)):
                        cls_precision_[classes[m]].append(cls_precision[m,j][idx[m]])
                        cls_recall_[classes[m]].append(cls_recall[m,j][idx[m]])

                logger.info("===============================================================\n")

            print(result_str)

            if 'PR_detail_dict' in result_dict:
                PR_detail_dict_3d = result_dict['PR_detail_dict']['3d']
                print("Plot Car PR Curve")
                precision = {
                    'easy':PR_detail_dict_3d['precision'][0,0,0],
                    'mod':PR_detail_dict_3d['precision'][0,1,0],
                    'hard':PR_detail_dict_3d['precision'][0,2,0]
                }
                plot_pr_curve(precision,
                              out_pic_fig_path = config_dir + "/result_eval/pr-curve.png")
                logger.info("\n==================== Precision & Recall Curve Data =================")
                str_precision = str(PR_detail_dict_3d['precision'][0,0,0])
                logger.info("\n car 3d precision R40@0.7 - easy:\n %s"%(str(PR_detail_dict_3d['precision'][0,0,0])))
                logger.info("\n car 3d precision R40@0.7 - mod:\n %s"%(str(PR_detail_dict_3d['precision'][0,1,0])))
                logger.info("\n car 3d precision R40@0.7 - hard:\n %s"%(str(PR_detail_dict_3d['precision'][0,2,0])))
                logger.info("\n pedestrian 3d precision R40@0.5 - easy:\n %s"%(str(PR_detail_dict_3d['precision'][1,0,0])))
                logger.info("\n pedestrian 3d precision R40@0.5 - mod:\n %s"%(str(PR_detail_dict_3d['precision'][1,1,0])))
                logger.info("\n pedestrian 3d precision R40@0.5 - hard:\n %s"%(str(PR_detail_dict_3d['precision'][1,2,0])))
                logger.info("\n cyclist 3d precision R40@0.5 - easy:\n %s"%(str(PR_detail_dict_3d['precision'][2,0,0])))
                logger.info("\n cyclist 3d precision R40@0.5 - mod:\n %s"%(str(PR_detail_dict_3d['precision'][2,1,0])))
                logger.info("\n cyclist 3d precision R40@0.5 - hard:\n %s"%(str(PR_detail_dict_3d['precision'][2,2,0])))

            # save result_dict
            with open(result_pkl_file.replace('result.pkl','result_dict.pkl'),"wb") as f:
                pickle.dump(result_dict,f)
            
            # elodie - ignore_class
            _, result_dict_ignore_class = dataset.evaluation(
                det_annos, dataset.class_names,
                eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
                ignore_classes=True
            )
            if 'min_thresh_ret' in result_dict_ignore_class:
                cls_recall = result_dict_ignore_class['min_thresh_ret']['recall_min_thresh']*100
                cls_precision = result_dict_ignore_class['min_thresh_ret']['precision_min_thresh']*100
                logger.info("\n==================== Ignore Class - Precision & Recall Result =================")
                overlap = np.array([[0.7,0.5,0.5],
                                    [0.5,0.25,0.25],
                                    [0.4,0.4,0.4],
                                    [0.2,0.1,0.1],
                                    [0.0,0.0,0.0]])
                idx = np.array([4,3,2,1,0])
                logger.info("              Easy     Mod      Hard")
                for i in idx:
                    logger.info("recall@%.1f:   %.2f    %.2f    %.2f"%(overlap[i,0], cls_recall[0,0,i], cls_recall[0,1,i], cls_recall[0,2,i]))
                    logger.info("precison@%.1f: %.2f    %.2f    %.2f \n"%(overlap[i,0],cls_precision[0,0,i], cls_precision[0,1,i], cls_precision[0,2,i]))
                for j in range(3):
                    cls_precision_['ignore_class'].append(cls_precision[0,j][idx])
                    cls_recall_['ignore_class'].append(cls_recall[0,j][idx])
                logger.info("===============================================================\n")

            out_pic_fig = config_dir + "/result_eval/pr.png"
            plot_recall_iou(cls_recall_, cls_precision_, overlap, out_pic_fig)              
            logger.info('pr picture save to %s'%out_pic_fig)

            logger.info('****************Evaluation done.*****************')

        if eval_type == "iou":
            det_w_iou = dataset.get_detobject_iou(det_annos)
            result_w_iou_pkl_file= result_pkl_file.replace('result.pkl', 'result_w_iou.pkl')
            with open(result_w_iou_pkl_file,"wb") as f:
                pickle.dump(det_w_iou,f)
        # print(result_str)
