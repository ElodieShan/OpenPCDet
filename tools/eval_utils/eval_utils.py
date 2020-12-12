import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, save_iou=False, use_sub_data=False):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        if use_sub_data and "16lines" in batch_dict:
            import copy            
            batch_dict_sub = copy.deepcopy(batch_dict)
            batch_dict_sub['points'] = batch_dict_sub['16lines']['points_16lines']
            batch_dict_sub['voxels'] = batch_dict_sub['16lines']['voxels']
            batch_dict_sub['voxel_coords'] = batch_dict_sub['16lines']['voxel_coords']
            batch_dict_sub['voxel_num_points'] = batch_dict_sub['16lines']['voxel_num_points']
            batch_dict_sub.pop('16lines')
            batch_dict.pop('16lines')
            load_data_to_gpu(batch_dict_sub)
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                pred_dicts, ret_dict = model(batch_dict, batch_dict_sub=batch_dict_sub)
        else:
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
            
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)
    # cls recall & precision
    logger.info("\n==================== Anchor Cls Result =================")
    logger.info("              Car     Pedestrian      Cyclist")
    logger.info("recall:   %.2f    %.2f        %.2f"%(ret_dict['cls_recall'][0], ret_dict['cls_recall'][1], ret_dict['cls_recall'][2]))
    logger.info("precison: %.2f    %.2f        %.2f "%(ret_dict['cls_precision'][0], ret_dict['cls_precision'][1], ret_dict['cls_precision'][2]))
    logger.info("=======================================================\n")

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f \n'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )
    
    if save_iou: #elodie
        det_annos_w_iou = dataset.get_detobject_iou(det_annos)
        with open(result_dir / 'result_w_iou.pkl', 'wb') as f:
            pickle.dump(det_annos_w_iou, f)

    if 'min_thresh_ret' in result_dict:
        cls_recall = result_dict['min_thresh_ret']['recall_min_thresh']*100
        cls_precision = result_dict['min_thresh_ret']['precision_min_thresh']*100
        logger.info("==================== Precision & Recall Result =================")
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
        logger.info("===============================================================\n")
        result_dict.pop('min_thresh_ret')
    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)

    # ignore_class
    _, result_dict_ignore_class = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir,
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
        logger.info("===============================================================\n")

    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
