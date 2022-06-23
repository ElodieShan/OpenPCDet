import torch
import numpy as np
import cv2

def class_score_filter(box_scores, box_preds, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
    return original_idxs, src_box_scores[original_idxs]

def nms_rotate_cpu(box_scores, box_preds, nms_config, score_thresh=None):
    """
    :param box_scores: scores of boxes
    :param box_preds: format [x, y, z, w, l, h, theta(rad)]
    :param nms_config: 
    :param score_thresh: 
    """
    rad_to_deg = 57.2957795
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]
    _, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
    original_idxs = scores_mask.nonzero().view(-1)
    keep_idx = []
    indices = list(indices)
    while len(indices)>0:
        idx = indices[0]
        j = 1
        tbox = box_preds[idx]
        tarea  = tbox[3] * tbox[4]
        while j < len(indices):
            idx_j = indices[j]
            box = box_preds[idx_j]
            area  = box[3] * box[4]
            try:
                int_pts = cv2.rotatedRectangleIntersection(((tbox[0], tbox[1]), (tbox[3], tbox[4]), tbox[6]*rad_to_deg), ((box[0], box[1]), (box[3], box[4]), box[6]*rad_to_deg))[1]
                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)
                    int_area  = cv2.contourArea(order_pts)
                    inter     = int_area * 1.0 / (tarea + area - int_area + 1e-5)  # compute IoU
                else:
                    inter = 0
            except:
                inter = 0.9999
            if inter > nms_config.NMS_THRESH:
                indices.pop(j)
            else:
                j += 1
                    
        keep_idx.append(indices.pop(0))
    keep_idx = np.array(keep_idx, np.int64)
    box_scores = box_scores[keep_idx]
    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        keep_idx = original_idxs[keep_idx]
    return keep_idx, box_scores